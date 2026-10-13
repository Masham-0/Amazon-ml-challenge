// Country-aware inverted retrieval and string feature extraction. C++17.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
constexpr int BASE_NF = 109, NF = 134;
using U = uint32_t;
using H = uint64_t;
H hashkey(const string &s, H seed = 1469598103934665603ULL) {
  for (unsigned char c : s) {
    seed ^= c;
    seed *= 1099511628211ULL;
  }
  return seed;
}
vector<string> split(const string &s, char delim = ' ') {
  vector<string> out;
  size_t p = 0, q;
  while ((q = s.find(delim, p)) != string::npos) {
    out.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  out.push_back(s.substr(p));
  return out;
}
vector<string> toks(const string &s) {
  auto t = split(s);
  t.erase(remove(t.begin(), t.end(), ""), t.end());
  sort(t.begin(), t.end());
  t.erase(unique(t.begin(), t.end()), t.end());
  return t;
}
string join(const vector<string> &t) {
  string s;
  for (auto &x : t) {
    if (s.size())
      s += ' ';
    s += x;
  }
  return s;
}
string phon(const string &s) {
  string o;
  char prev = 0;
  for (char c : s) {
    if (c == ' ') {
      prev = 0;
      if (!o.empty() && o.back() != ' ')
        o += ' ';
      continue;
    }
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue;
    if (c == 'v')
      c = 'w';
    if (c == 'q')
      c = 'k';
    if (c == 'z')
      c = 's';
    if (c != prev) {
      o += c;
      prev = c;
    }
  }
  return o;
}
vector<string> nums(const vector<string> &v) {
  vector<string> o;
  for (auto &t : v)
    if (all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; }))
      o.push_back(t);
  return o;
}
float frac(int n, int d) { return d ? float(n) / d : 0; }
int intersection(const vector<string> &a, const vector<string> &b) {
  int n = 0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      n++;
      i++;
      j++;
    } else if (a[i] < b[j])
      i++;
    else
      j++;
  }
  return n;
}
float lcs(const string &a, const string &b) {
  if (a.empty() || b.empty())
    return 0;
  if (a == b)
    return 1;
  if (min(a.size(), b.size()) <= 64) {
    const string &x = a.size() < b.size() ? a : b;
    const string &y = a.size() < b.size() ? b : a;
    H masks[128] = {}, state = 0;
    for (size_t i = 0; i < x.size(); i++)
      masks[(unsigned char)x[i]] |= H(1) << i;
    for (unsigned char c : y) {
      H v = state | masks[c];
      state = v & ~(v - ((state << 1) | 1));
    }
    return 2.f * __builtin_popcountll(state) / (a.size() + b.size());
  }
  vector<int> dp(b.size() + 1);
  for (char ca : a) {
    int prev = 0;
    for (size_t j = 0; j < b.size(); j++) {
      int old = dp[j + 1];
      dp[j + 1] = ca == b[j] ? prev + 1 : max(dp[j], dp[j + 1]);
      prev = old;
    }
  }
  return 2.f * dp.back() / (a.size() + b.size());
}
vector<H> grams(const string &s, int n = 3) {
  vector<H> o;
  if (s.size() >= size_t(n))
    for (size_t i = 0; i + n <= s.size(); i++)
      o.push_back(hashkey(s.substr(i, n)));
  sort(o.begin(), o.end());
  o.erase(unique(o.begin(), o.end()), o.end());
  return o;
}
float gdice(const vector<H> &a, const vector<H> &b) {
  size_t i = 0, j = 0;
  int n = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      n++;
      i++;
      j++;
    } else if (a[i] < b[j])
      i++;
    else
      j++;
  }
  return frac(2 * n, a.size() + b.size());
}
unordered_map<string, string> aliases;
string alias_name(const string &n, bool address = false) {
  string o;
  for (auto &t : split(n)) {
    auto i = aliases.find((address ? "@" : "") + t);
    string v = i == aliases.end() ? t : i->second;
    if (v == "-" || v.empty())
      continue;
    if (o.size())
      o += ' ';
    o += v;
  }
  return o;
}
string compact(string s) {
  s.erase(remove(s.begin(), s.end(), ' '), s.end());
  return s;
}
bool country_matches(const string &line, const string &country) {
  if (country.empty())
    return true;
  size_t a = line.find('\t'), b = line.find('\t', a + 1);
  return a != string::npos && b != string::npos &&
         line.compare(a + 1, b - a - 1, country) == 0;
}
struct Rec {
  U id;
  string country, n, a, sn, sa, pn, rn, srn, firstn, firstnum;
  vector<string> nt, at, nu, rnt, numberwords, digits, floors, units,
      directions;
  vector<H> ng, ag;
};
Rec parse(const string &line) {
  auto x = split(line, '\t');
  if (x.size() < 4 || x.size() > 5)
    throw runtime_error("bad normalized row");
  Rec r;
  r.id = stoul(x[0].substr(3));
  if (x[0][1] == '3')
    r.id |= 0x80000000U;
  r.country = x[1];
  r.n = alias_name(x[2]);
  r.a = alias_name(x[3], true);
  r.rn = x.size() == 5 ? x[4] : x[2];
  r.rnt = toks(r.rn);
  r.srn = join(r.rnt);
  r.firstn = split(r.n)[0];
  r.nt = toks(r.n);
  r.at = toks(r.a);
  r.nu = nums(r.at);
  r.sn = join(r.nt);
  r.sa = join(r.at);
  r.pn = phon(r.sn);
  r.ng = grams(r.sn);
  r.ag = grams(r.sa);
  auto aw = split(r.a);
  for (size_t i = 0; i < aw.size(); i++) {
    auto &t = aw[i];
    bool hasnum = false;
    for (size_t j = 0; j < t.size(); j++)
      if (isdigit(t[j])) {
        size_t end = j;
        while (end < t.size() && isdigit(t[end]))
          end++;
        string d = t.substr(j, end - j);
        while (d.size() > 1 && d[0] == '0')
          d.erase(d.begin());
        r.digits.push_back(d);
        hasnum = true;
        j = end - 1;
      }
    if (hasnum) {
      r.numberwords.push_back(t);
      if (r.firstnum.empty())
        r.firstnum = t;
    }
    bool floor = t == "fl" || t == "flr" || t == "floor",
         unit = t == "unit" || t == "apt" || t == "ste" || t == "flat" ||
                t == "suite";
    if (floor || unit) {
      auto &dst = floor ? r.floors : r.units;
      for (int delta :
           floor ? vector<int>{-1, 1, -2, 2} : vector<int>{1, 2, -1}) {
        int j = int(i) + delta;
        if (j < 0 || j >= int(aw.size()))
          continue;
        string v = aw[j];
        if (v == "unit" || v == "apt" || v == "ste" || v == "flat")
          continue;
        if (v == "first")
          v = "1st";
        if (v == "second")
          v = "2nd";
        if (v == "third")
          v = "3rd";
        if (v == "ground" || v == "grd" || v == "gf")
          v = "0";
        if (any_of(v.begin(), v.end(), [](char c) { return isdigit(c); }) ||
            (unit && v.size() <= 6)) {
          if (floor) {
            string digits;
            for (char c : v)
              if (isdigit(c))
                digits += c;
            v = digits;
          }
          dst.push_back(v);
          break;
        }
      }
    }
  }
  sort(r.numberwords.begin(), r.numberwords.end());
  sort(r.digits.begin(), r.digits.end());
  sort(r.floors.begin(), r.floors.end());
  sort(r.units.begin(), r.units.end());
  r.floors.erase(unique(r.floors.begin(), r.floors.end()), r.floors.end());
  r.units.erase(unique(r.units.begin(), r.units.end()), r.units.end());
  for (auto &t : r.rnt)
    if (t == "north" || t == "south" || t == "east" || t == "west" ||
        t == "downtown" || t == "central" || t == "branch")
      r.directions.push_back(t);
  return r;
}
struct Key {
  H key;
  int type;
};
vector<Key> keys(const Rec &r) {
  vector<Key> o;
  H base = hashkey(r.country);
  auto add = [&](int type, const string &s) {
    if (!s.empty())
      o.push_back({hashkey(s, base ^ (H(type) * 0x9e3779b97f4a7c15ULL)), type});
  };
  add(0, r.sn);
#ifdef WIDE_BLOCKING
  auto whole = aliases.find("%" + r.sn);
  add(11, whole == aliases.end() ? r.sn : whole->second);
  string address = compact(r.a);
  for (size_t i = 0; i + 3 <= address.size(); i++)
    add(12, address.substr(i, 3));
#endif
  add(1, r.sa);
  add(6, compact(r.n));
  for (size_t i = 0; i < r.nt.size() && i < 8; i++)
    for (size_t j = i + 1; j < r.nt.size() && j < 8; j++)
      if (r.nt[i].size() >= 3 && r.nt[j].size() >= 3)
        add(7, r.nt[i] + "|" + r.nt[j]);
  int un = 0;
  for (auto &u : r.nu) {
    if (u.size() < 2 || ++un > 5)
      continue;
    int wn = 0;
    for (auto &w : r.at) {
      if (w.size() < 4 ||
          all_of(w.begin(), w.end(), [](char c) { return isdigit(c); }) ||
          ++wn > 12)
        continue;
      add(8, u + "|" + w);
    }
  }
  auto aw = split(r.a);
  for (size_t i = 1; i < aw.size(); i++)
    if (aw[i - 1].size() >= 3 && aw[i].size() >= 3)
      add(9, aw[i - 1] + "|" + aw[i]);
  for (auto &t : r.nt)
    if (t.size() >= 2)
      add(2, t);
  for (auto &t : r.at)
    if (t.size() >= 3)
      add(3, t);
  for (auto &t : r.at)
    if (t.size() >= 4 &&
        !any_of(t.begin(), t.end(), [](char c) { return isdigit(c); })) {
      string p = phon(t);
      if (p.size() >= 3)
        add(10, p);
    }
  for (auto &t : r.nt)
    if (t.size() >= 4) {
      string p = phon(t);
      if (p.size() >= 2)
        add(4, p);
    }
  string compact = r.sn;
  compact.erase(remove(compact.begin(), compact.end(), ' '), compact.end());
  for (size_t i = 0; i + 3 <= compact.size(); i++)
    add(5, compact.substr(i, 3));
  sort(o.begin(), o.end(), [](auto &a, auto &b) { return a.key < b.key; });
  o.erase(unique(o.begin(), o.end(),
                 [](auto &a, auto &b) { return a.key == b.key; }),
          o.end());
  return o;
}
unordered_map<H, vector<U>> *feature_index = nullptr;
size_t reference_count = 0;
float tokenweight(const Rec &r, const string &t, int type) {
  H key = hashkey(t, hashkey(r.country) ^ (H(type) * 0x9e3779b97f4a7c15ULL));
  auto it = feature_index->find(key);
  return it == feature_index->end()
             ? 1.f
             : log1p(double(reference_count) / it->second.size());
}
float nameweight(const Rec &r, const string &t) { return tokenweight(r, t, 2); }
float frequency(const Rec &r, const string &s, int type) {
  auto it = feature_index->find(
      hashkey(s, hashkey(r.country) ^ (H(type) * 0x9e3779b97f4a7c15ULL)));
  return it == feature_index->end() ? 0 : it->second.size();
}
array<float, NF> features(const Rec &a, const Rec &b, float retrieval,
                          float rank, float gap) {
  array<float, NF> f{};
  int k = 0;
  auto push = [&](float x) { f[k++] = x; };
  int ni = intersection(a.nt, b.nt), ai = intersection(a.at, b.at),
      ui = intersection(a.nu, b.nu);
  push(a.n == b.n && !a.n.empty());
  push(lcs(a.n, b.n));
  push(lcs(a.sn, b.sn));
  push(frac(ni, a.nt.size() + b.nt.size() - ni));
  push(frac(2 * ni, a.nt.size() + b.nt.size()));
  push(frac(ni, min(a.nt.size(), b.nt.size())));
  push(frac(ni, max(a.nt.size(), b.nt.size())));
  push(gdice(a.ng, b.ng));
  push(lcs(a.pn, b.pn));
  push(frac(min(a.n.size(), b.n.size()), max(a.n.size(), b.n.size())));
  push(ni);
  push(a.nt.size());
  push(b.nt.size());
  push(a.sa == b.sa && !a.sa.empty());
  push(frac(ai, a.at.size() + b.at.size() - ai));
  push(frac(2 * ai, a.at.size() + b.at.size()));
  push(frac(ai, min(a.at.size(), b.at.size())));
  push(frac(ai, max(a.at.size(), b.at.size())));
  push(gdice(a.ag, b.ag));
  push(lcs(a.sa, b.sa));
  push(ai);
  push(a.at.size());
  push(b.at.size());
  push(frac(ui, a.nu.size() + b.nu.size() - ui));
  push(frac(ui, min(a.nu.size(), b.nu.size())));
  push(a.nu == b.nu && !a.nu.empty());
  push(a.nu.size());
  push(b.nu.size());
  push(a.nu.empty() || b.nu.empty());
  push(a.n.empty() || b.n.empty());
  push(a.a.empty() || b.a.empty());
  push(retrieval);
  push(rank);
  push(gap);
  push((a.id >> 31) == (b.id >> 31));
  push(a.n.size() + b.n.size());
  int ri = intersection(a.rnt, b.rnt);
  push(a.rn == b.rn && !a.rn.empty());
  push(lcs(a.rn, b.rn));
  push(lcs(a.srn, b.srn));
  push(frac(ri, a.rnt.size() + b.rnt.size() - ri));
  push(frac(2 * ri, a.rnt.size() + b.rnt.size()));
  push(frac(ri, min(a.rnt.size(), b.rnt.size())));
  push(frac(ri, max(a.rnt.size(), b.rnt.size())));
  push(frac(min(a.rn.size(), b.rn.size()), max(a.rn.size(), b.rn.size())));
  push(b.rnt.size() - ri);
  push(a.rnt.size() - ri);
  vector<float> fa(a.nt.size(), 0), fb(b.nt.size(), 0);
  for (size_t i = 0; i < a.nt.size(); i++)
    for (size_t j = 0; j < b.nt.size(); j++) {
      float sim = lcs(a.nt[i], b.nt[j]);
      fa[i] = max(fa[i], sim);
      fb[j] = max(fb[j], sim);
    }
  float ma = fa.empty() ? 0 : accumulate(fa.begin(), fa.end(), 0.f) / fa.size(),
        mb = fb.empty() ? 0 : accumulate(fb.begin(), fb.end(), 0.f) / fb.size();
  push(min(ma, mb));
  push(max(ma, mb));
  push(fa.empty() ? 0 : *min_element(fa.begin(), fa.end()));
  push(fb.empty() ? 0 : *min_element(fb.begin(), fb.end()));
  push(lcs(a.firstn, b.firstn));
  push(lcs(phon(a.firstn), phon(b.firstn)));
  float wa = 0, wb = 0, shared = 0;
  for (auto &t : a.nt) {
    float w = nameweight(a, t);
    wa += w;
    if (binary_search(b.nt.begin(), b.nt.end(), t))
      shared += w;
  }
  for (auto &t : b.nt)
    wb += nameweight(b, t);
  push((wa + wb) > 0 ? 2 * shared / (wa + wb) : 0);
  push(wa > 0 ? (wa - shared) / wa : 0);
  push(wb > 0 ? (wb - shared) / wb : 0);
  int mi = intersection(a.numberwords, b.numberwords);
  push(frac(mi, a.numberwords.size() + b.numberwords.size() - mi));
  push(frac(2 * mi, a.numberwords.size() + b.numberwords.size()));
  push(frac(mi, min(a.numberwords.size(), b.numberwords.size())));
  push(frac(mi, max(a.numberwords.size(), b.numberwords.size())));
  push(a.numberwords == b.numberwords && !a.numberwords.empty());
  push(a.numberwords.size());
  push(b.numberwords.size());
  push(a.numberwords.size() - mi);
  push(b.numberwords.size() - mi);
  push(a.firstnum == b.firstnum && !a.firstnum.empty());
  push(lcs(a.firstnum, b.firstnum));
  int di = intersection(a.digits, b.digits);
  push(frac(di, a.digits.size() + b.digits.size() - di));
  push(frac(2 * di, a.digits.size() + b.digits.size()));
  push(a.digits == b.digits && !a.digits.empty());
  push(a.digits.size() - di);
  push(b.digits.size() - di);
  push(a.floors == b.floors && !a.floors.empty());
  push(!a.floors.empty() && !b.floors.empty() && a.floors != b.floors);
  push(a.floors.empty() || b.floors.empty());
  push(a.units == b.units && !a.units.empty());
  push(!a.units.empty() && !b.units.empty() && a.units != b.units);
  push(a.units.empty() || b.units.empty());
  push(a.directions != b.directions);
  push(b.directions.size() - intersection(a.directions, b.directions));
  bool zeros = false;
  for (auto &t : b.numberwords)
    if (t.size() > 1 && t[0] == '0')
      zeros = true;
  push(zeros);
  string an = compact(a.sn), bn = compact(b.sn);
  float sim = lcs(an, bn);
  push(sim);
  push(min(an.size(), bn.size())
           ? sim * (an.size() + bn.size()) / (2 * min(an.size(), bn.size()))
           : 0);
  string ia, ib, ica, icb;
  vector<string> ca, cb;
  for (auto &t : split(a.n)) {
    if (t.empty())
      continue;
    ia += t[0];
    if (nameweight(a, t) > log1p(100.)) {
      ica += t[0];
      ca.push_back(t);
    }
  }
  for (auto &t : split(b.n)) {
    if (t.empty())
      continue;
    ib += t[0];
    if (nameweight(b, t) > log1p(100.)) {
      icb += t[0];
      cb.push_back(t);
    }
  }
  push((an == ib && !ib.empty()) || (bn == ia && !ia.empty()));
  push((an == icb && !icb.empty()) || (bn == ica && !ica.empty()));
  push(frequency(a, a.sn, 0));
  push(frequency(b, b.sn, 0));
  sort(ca.begin(), ca.end());
  sort(cb.begin(), cb.end());
  int ci = intersection(ca, cb);
  push(frac(ci, ca.size() + cb.size() - ci));
  push(ca == cb && !ca.empty());
  push(ca.size());
  push(cb.size());
  push(cb.size() - ci);
  push(ca.size() - ci);
  string af = split(a.rn)[0], bf = split(b.rn)[0];
  push(lcs(af, bf));
  push(af == bf && !af.empty());
  float aa = 0, bb = 0, ab = 0, rare = 0;
  ca.clear();
  cb.clear();
  for (auto &t : a.at) {
    float w = tokenweight(a, t, 3);
    aa += w;
    if (binary_search(b.at.begin(), b.at.end(), t)) {
      ab += w;
      rare = max(rare, w);
    }
    if (w > log1p(100.))
      ca.push_back(t);
  }
  for (auto &t : b.at) {
    float w = tokenweight(b, t, 3);
    bb += w;
    if (w > log1p(100.))
      cb.push_back(t);
  }
  push((aa + bb - ab) > 0 ? ab / (aa + bb - ab) : 0);
  push((aa + bb) > 0 ? 2 * ab / (aa + bb) : 0);
  push(aa > 0 ? ab / aa : 0);
  push(bb > 0 ? ab / bb : 0);
  push(rare);
  ci = intersection(ca, cb);
  push(frac(ci, ca.size() + cb.size() - ci));
  push(ca == cb && !ca.empty());
  push(ca.size());
  push(cb.size());
  vector<float> numberbest;
  int deletion = 0, substitution = 0, transposition = 0;
  for (auto &t : a.numberwords) {
    float best = 0;
    string u;
    for (auto &v : b.numberwords) {
      float s = lcs(t, v);
      if (s > best) {
        best = s;
        u = v;
      }
    }
    numberbest.push_back(best);
    if (t == u || u.empty())
      continue;
    int common = int(round(best * (t.size() + u.size()) / 2));
    if (common == int(min(t.size(), u.size())))
      deletion++;
    if (t.size() == u.size()) {
      int changes = 0;
      for (size_t j = 0; j < t.size(); j++)
        changes += t[j] != u[j];
      if (changes <= 2)
        substitution++;
      string p = t, q = u;
      sort(p.begin(), p.end());
      sort(q.begin(), q.end());
      if (p == q)
        transposition++;
    }
  }
  push(numberbest.empty()
           ? 0
           : accumulate(numberbest.begin(), numberbest.end(), 0.f) /
                 numberbest.size());
  push(numberbest.empty() ? 0
                          : *min_element(numberbest.begin(), numberbest.end()));
  string da, db;
  for (auto &t : a.digits)
    da += t;
  for (auto &t : b.digits)
    db += t;
  sort(da.begin(), da.end());
  sort(db.begin(), db.end());
  push(da == db && !da.empty());
  push(deletion);
  push(substitution);
  push(transposition);
  if (k != BASE_NF)
    throw runtime_error("feature count");
  return f;
}
struct Row {
  U ref, target;
  array<float, NF> f;
};
struct Scored {
  U ref, target;
  float p;
};
static_assert(sizeof(Row) == (2 + NF) * sizeof(float), "Feature file layout");
static_assert(sizeof(Scored) == 12, "Score file layout");
struct LightGBM {
  void *lib = nullptr;
  void *model = nullptr;
  int (*predict)(void *, const void *, int, int32_t, int32_t, int, int, int,
                 int, const char *, int64_t *, double *) = nullptr;
  LightGBM(const string &libpath, const string &modelpath) {
    lib = dlopen(libpath.c_str(), RTLD_LAZY);
    if (!lib)
      throw runtime_error(dlerror());
    auto load = (int (*)(const char *, int *, void **))dlsym(
        lib, "LGBM_BoosterCreateFromModelfile");
    predict = (decltype(predict))dlsym(lib, "LGBM_BoosterPredictForMat");
    int n;
    if (!load || !predict || load(modelpath.c_str(), &n, &model))
      throw runtime_error("LightGBM loading failed");
  }
  vector<double> score(const vector<Row> &rows) {
    vector<float> x;
    x.reserve(rows.size() * NF);
    for (auto &r : rows)
      x.insert(x.end(), r.f.begin(), r.f.end());
    return score_matrix(x, NF);
  }
  vector<double> score_matrix(const vector<float> &x, int width) {
    vector<double> p(x.size() / width);
    int64_t n;
    int err = predict(model, x.data(), 0, p.size(), width, 1, 0, 0, -1,
                      "num_threads=6", &n, p.data());
    if (err || n != int64_t(p.size()))
      throw runtime_error("prediction failed");
    return p;
  }
};
int main(int argc, char **argv) {
  try {
    if (argc < 6) {
      cerr << "engine REFS TARGET2 TARGET3 OUT_PREFIX sample_truth|- [model "
              "library] [gate=0.30] [gap=0.25]\n";
      return 1;
    }
    if (argc > 10) {
      ifstream af(argv[10]);
      string al;
      while (getline(af, al)) {
        auto t = split(al, '\t');
        if (t.size() == 2)
          aliases[t[0]] = t[1];
      }
      cerr << "aliases " << aliases.size() << endl;
    }
    auto start = chrono::steady_clock::now();
    string line, prefix = argv[4], country = argc > 11 ? argv[11] : "";
    bool train = string(argv[5]) != "-";
    float gate = argc > 8 ? stof(argv[8]) : .30f,
          keepgap = argc > 9 ? stof(argv[9]) : .25f;
    int candidate_limit = argc > 13 ? stoi(argv[13]) : 6;
    int name_only_limit = argc > 14 ? stoi(argv[14]) : candidate_limit;
    float name_only_gap = argc > 15 ? stof(argv[15]) : keepgap;
    if (candidate_limit < 1 || candidate_limit > 100 || name_only_limit < 1 ||
        name_only_limit > 100 || name_only_gap < 0 || name_only_gap > 1)
      throw runtime_error("invalid candidate limits");
    unordered_set<U> sample;
    unordered_map<U, U> trueowner;
    if (train) {
      ifstream gt(argv[5]);
      while (getline(gt, line)) {
        auto x = split(line, '\t');
        U id = stoul(x[0].substr(3));
        sample.insert(id);
        if (x.size() > 1 && !x[1].empty())
          for (auto &s : split(x[1], ',')) {
            U t = stoul(s.substr(3));
            if (s[1] == '3')
              t |= 0x80000000U;
            trueowner[t] = id;
          }
      }
    }
    vector<Rec> refs;
    refs.reserve(2300000);
    unordered_map<U, U> id_to_row;
    unordered_map<H, vector<U>> index;
    index.reserve(6000000);
    ifstream inp(argv[1]);
    ofstream refout(prefix + ".refs.tsv");
    while (getline(inp, line)) {
      if (!country_matches(line, country))
        continue;
      auto r = parse(line);
      U i = refs.size();
      if (sample.count(r.id))
        id_to_row[r.id] = i;
      refout << "S1-" << r.id << '\t' << r.country << '\t' << sample.count(r.id)
             << '\n';
      for (auto &k : keys(r))
        index[k.key].push_back(i);
      refs.push_back(move(r));
      if (refs.size() % 500000 == 0)
        cerr << "indexed " << refs.size() << " keys " << index.size() << endl;
    }
    refout.close();
    feature_index = &index;
    reference_count = refs.size();
    cerr << "index ready " << refs.size() << " / " << index.size() << endl;
    struct State {
      vector<float> acc;
      vector<U> stamp, touched;
      U epoch = 0;
      State(size_t n) : acc(n, 0), stamp(n, 0) {}
    };
    vector<State> states;
    for (int i = 0; i < 4; i++)
      states.emplace_back(refs.size());
    uint64_t nq = 0, np = 0, ntruth = 0, nfound = 0;
    ofstream out(prefix + (train ? ".features.bin" : ".scores.bin"),
                 ios::binary),
        missing(prefix + ".missing.tsv");
    vector<unique_ptr<LightGBM>> models;
    unique_ptr<ofstream> feature_dump;
    if (!train) {
      if (argc < 8)
        throw runtime_error("model and shared library required");
      for (auto &path : split(argv[6], ','))
        models.push_back(make_unique<LightGBM>(argv[7], path));
      if (argc > 12 && string(argv[12]) == "dump")
        feature_dump =
            make_unique<ofstream>(prefix + ".features.bin", ios::binary);
    }
    vector<Row> batch;
    batch.reserve(65536);
    auto flush = [&]() {
      if (batch.empty())
        return;
      if (train) {
        out.write(reinterpret_cast<const char *>(batch.data()),
                  batch.size() * sizeof(Row));
      } else {
        if (feature_dump)
          feature_dump->write(reinterpret_cast<const char *>(batch.data()),
                              batch.size() * sizeof(Row));
        vector<double> p(batch.size(), 0);
        for (auto &model : models) {
          auto pred = model->score(batch);
          for (size_t j = 0; j < p.size(); j++)
            p[j] += pred[j] / models.size();
        }
        for (size_t j = 0; j < batch.size(); j++) {
          Scored s{batch[j].ref, batch[j].target, float(p[j])};
          out.write(reinterpret_cast<const char *>(&s), sizeof(s));
        }
      }
      batch.clear();
    };
    struct Result {
      vector<Row> rows;
      int truth = 0, found = 0;
      ostringstream missing;
    };
    auto process = [&](const string &line, State &state) {
      Result result;
      auto &acc = state.acc;
      auto &stamp = state.stamp;
      auto &touched = state.touched;
      auto &epoch = state.epoch;
      Rec q = parse(line);
      ++epoch;
      touched.clear();
      auto qkeys = keys(q);
      struct Block {
        const vector<U> *p;
        int type;
      };
      vector<Block> blocks;
      for (auto &k : qkeys) {
        auto it = index.find(k.key);
        if (it == index.end())
          continue;
        int count = it->second.size();
        if (count > 5000)
          continue;
        blocks.push_back({&it->second, k.type});
      }
      sort(blocks.begin(), blocks.end(), [](auto &a, auto &b) {
        if (a.type < 2 || b.type < 2) {
          if (a.type != b.type)
            return a.type < b.type;
        }
        return a.p->size() < b.p->size();
      });
      int used[13] = {}, budget = 0;
#ifdef WIDE_BLOCKING
      constexpr int query_budget = 3000;
#else
      constexpr int query_budget = 1500;
#endif
      for (auto &b : blocks) {
        int maxuse = b.type == 5 ? 12 : b.type >= 7 ? 4 : 3;
#ifdef WIDE_BLOCKING
        maxuse = b.type == 12   ? 24
                 : b.type == 11 ? 3
                 : b.type == 5  ? 16
                 : b.type >= 7  ? 4
                                : 3;
#endif
        if (used[b.type] >= maxuse)
          continue;
        if (b.type >= 2 && budget + int(b.p->size()) > query_budget &&
            used[b.type])
          continue;
        if (b.type >= 2 && b.p->size() > query_budget)
          continue;
        used[b.type]++;
        budget += b.p->size();
        float w = (b.type == 0                   ? 14.f
                   : b.type == 1                 ? 16.f
                   : b.type == 11                ? 14.f
                   : b.type == 12                ? .4f
                   : b.type == 5                 ? .5f
                   : b.type == 4 || b.type == 10 ? 1.0f
                                                 : 2.f) *
                  log1p(double(refs.size()) / b.p->size());
        for (U i : *b.p) {
          if (stamp[i] != epoch) {
            stamp[i] = epoch;
            acc[i] = w;
            touched.push_back(i);
          } else
            acc[i] += w;
        }
      }
      vector<pair<float, U>> ranked;
      ranked.reserve(touched.size());
      for (U i : touched)
        ranked.emplace_back(acc[i], i);
      int maxrough = 100;
#ifdef WIDE_BLOCKING
      maxrough = 200;
#endif
      if (ranked.size() > size_t(maxrough)) {
        nth_element(ranked.begin(), ranked.begin() + maxrough, ranked.end(),
                    greater<>());
        ranked.resize(maxrough);
      }
      sort(ranked.begin(), ranked.end(), greater<>());
      struct Cand {
        U i;
        float cheap, ret;
      };
      vector<Cand> cs;
      cs.reserve(ranked.size());
      for (auto &v : ranked) {
        const Rec &r = refs[v.second];
        float ns = lcs(q.sn, r.sn), as = gdice(q.ag, r.ag),
              nc =
                  frac(intersection(q.nt, r.nt), min(q.nt.size(), r.nt.size())),
              ac =
                  frac(intersection(q.at, r.at), min(q.at.size(), r.at.size()));
        ns = max(ns, .90f * lcs(q.pn, r.pn));
#ifdef WIDE_BLOCKING
        auto qa = aliases.find("%" + q.sn), ra = aliases.find("%" + r.sn);
        if (qa != aliases.end() || ra != aliases.end())
          ns = max(ns, lcs(qa == aliases.end() ? q.sn : qa->second,
                           ra == aliases.end() ? r.sn : ra->second));
#endif
        float ad = max(as, .88f * ac);
        float cheap =
            max(max(ns, .90f * nc) *
                    (q.a.empty() || r.a.empty() ? .72f : .42f + .58f * ad),
                .84f * ad);
        cs.push_back({v.second, cheap, v.first});
      }
      sort(cs.begin(), cs.end(),
           [](auto &a, auto &b) { return a.cheap > b.cheap; });
      bool found = false;
      U ti = 0;
      auto truth = trueowner.find(q.id);
      if (train && truth != trueowner.end()) {
        ti = id_to_row.at(truth->second);
        result.truth = 1;
      }
      int kept = 0;
      vector<size_t> chosen;
      bool relevant = !train;
      int limit = q.a.empty() ? name_only_limit : candidate_limit;
      float gap = q.a.empty() ? name_only_gap : keepgap;
      for (size_t j = 0; j < cs.size() && kept < limit; j++) {
        auto &c = cs[j];
        if (c.cheap < gate || c.cheap < cs[0].cheap - gap)
          continue;
        kept++;
        if (train && truth != trueowner.end() && c.i == ti)
          found = true;
        chosen.push_back(j);
        if (sample.count(refs[c.i].id))
          relevant = true;
      }
      // Retain every competitor for targets touching sampled references.
      // Only sampled reference labels enter fitting; all candidates enter
      // validation ownership decisions, exactly as in final inference.
      if (relevant)
        for (size_t j : chosen) {
          auto &c = cs[j];
          Row row{c.i, q.id,
                  features(refs[c.i], q, c.ret, float(j),
                           c.cheap -
                               (cs.size() > 1 ? cs[j == 0 ? 1 : 0].cheap : 0))};
          result.rows.push_back(row);
        }
      // The eight views are compared only against other final candidates.
      array<int, 8> component = {2, 18, 5, 16, 37, 52, 55, 2};
      vector<array<float, 8>> views(result.rows.size());
      for (size_t j = 0; j < result.rows.size(); j++) {
        auto &f = result.rows[j].f;
        float ns = max(f[2], .9f * f[8]), ad = max(f[18], .88f * f[16]);
        views[j] = {ns,    ad,    f[5],  f[16],
                    f[37], f[52], f[55], ns * (.42f + .58f * ad)};
      }
      for (size_t j = 0; j < result.rows.size(); j++) {
        auto &f = result.rows[j].f;
        f[BASE_NF] = result.rows.size() - 1;
        for (int c = 0; c < 8; c++) {
          float other = 0, best = views[j][c];
          int close = 0;
          for (size_t t = 0; t < views.size(); t++)
            best = max(best, views[t][c]);
          for (size_t t = 0; t < views.size(); t++)
            if (t != j) {
              other = max(other, views[t][c]);
              close += views[t][c] >= best - .05f;
            }
          f[BASE_NF + 1 + c] = other;
          f[BASE_NF + 9 + c] = views[j][c] - other;
          f[BASE_NF + 17 + c] = close;
        }
      }
      if (train && truth != trueowner.end()) {
        if (found)
          result.found = 1;
        else
          result.missing << "S1-" << truth->second << '\t'
                         << (q.id >> 31 ? "S3-" : "S2-") << (q.id & 0x7fffffffU)
                         << '\t' << (cs.empty() ? 0 : cs[0].cheap) << '\t'
                         << q.n << '\t' << q.a << '\n';
      }
      return result;
    };
    for (int file = 2; file <= 3; file++) {
      ifstream targets(argv[file]);
      bool finished = false;
      uint64_t nextlog = 500000;
      while (!finished) {
        vector<string> lines;
        lines.reserve(8192);
        while (lines.size() < 8192) {
          if (!getline(targets, line)) {
            finished = true;
            break;
          }
          if (!country_matches(line, country))
            continue;
          lines.push_back(line);
        }
        vector<Result> results(lines.size());
#pragma omp parallel for num_threads(4) schedule(dynamic, 64)
        for (size_t j = 0; j < lines.size(); j++)
          results[j] = process(lines[j], states[omp_get_thread_num()]);
        for (auto &r : results) {
          nq++;
          ntruth += r.truth;
          nfound += r.found;
          missing << r.missing.str();
          for (auto &row : r.rows) {
            batch.push_back(row);
            np++;
            if (batch.size() >= 65536)
              flush();
          }
        }
        if (nq >= nextlog) {
          cerr << "queries " << nq << " pairs " << np << " sampled recall "
               << (ntruth ? double(nfound) / ntruth : 0) << " seconds "
               << chrono::duration<double>(chrono::steady_clock::now() - start)
                      .count()
               << endl;
          out.flush();
          nextlog = ((nq / 500000) + 1) * 500000;
        }
      }
    }
    flush();
    cerr
        << "DONE queries " << nq << " pairs " << np << " recall "
        << (ntruth ? double(nfound) / ntruth : 0) << " seconds "
        << chrono::duration<double>(chrono::steady_clock::now() - start).count()
        << endl;
  } catch (const exception &e) {
    cerr << e.what() << endl;
    return 2;
  }
  return 0;
}
