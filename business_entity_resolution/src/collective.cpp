// Leave-one-record-out support from confidently matched related records.
#define main retrieval_main
#include "engine.cpp"
#undef main
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(COPY_SUPPORT) && !defined(EXTRA_TFIDF)
#error COPY_SUPPORT requires EXTRA_TFIDF
#endif
#ifdef COPY_SUPPORT
constexpr int SUPPORT_NF = 36;
#elif defined(EXTRA_TFIDF)
constexpr int SUPPORT_NF = 32;
#else
constexpr int SUPPORT_NF = 26;
#endif
constexpr int SECOND_NF = NF + SUPPORT_NF;
template <class T> struct Mapping {
  int fd;
  size_t bytes, n;
  const T *data;
  Mapping(const string &path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    bytes = st.st_size;
    if (bytes % sizeof(T))
      throw runtime_error("unaligned file " + path);
    n = bytes / sizeof(T);
    data = (const T *)mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED)
      throw runtime_error("mapping failed");
  }
  ~Mapping() {
    munmap((void *)data, bytes);
    close(fd);
  }
  const T &operator[](size_t j) const { return data[j]; }
};
struct SupportRec {
  string name, compactname, address;
  vector<string> numbers;
  vector<H> ag;
#ifdef COPY_SUPPORT
  H signature = 0;
#endif
};
#ifdef EXTRA_TFIDF
struct TextRec {
  H country;
  vector<H> ng, ag, nw, aw;
};
unordered_map<H, U> text_df;
unordered_map<H, U> country_size;
TextRec text_parse(const string &line) {
  auto x = split(line, '\t');
  string name = alias_name(x[2]), address = alias_name(x[3], true);
  TextRec r;
  r.country = hashkey(x[1]);
  r.ng = grams(compact(join(toks(name))));
  r.ag = grams(join(toks(address)));
  for (auto &t : toks(name))
    r.nw.push_back(hashkey(t));
  for (auto &t : toks(address))
    r.aw.push_back(hashkey(t));
  sort(r.nw.begin(), r.nw.end());
  sort(r.aw.begin(), r.aw.end());
  return r;
}
H text_key(H gram, H country, int type) {
  return gram ^ country ^ (H(type + 101) * 0x9e3779b97f4a7c15ULL);
}
float text_weight(H gram, H country, int type) {
  auto i = text_df.find(text_key(gram, country, type));
  return log1p(double(country_size.at(country)) /
               max(1U, i == text_df.end() ? 0U : i->second));
}
array<float, 3> text_similarity(const vector<H> &a, const vector<H> &b,
                                H country, int type) {
  double aa = 0, bb = 0, ab = 0, wa = 0, wb = 0, shared = 0;
  for (H t : a) {
    double w = text_weight(t, country, type);
    aa += w * w;
    wa += w;
  }
  for (H t : b) {
    double w = text_weight(t, country, type);
    bb += w * w;
    wb += w;
  }
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j])
      i++;
    else if (a[i] > b[j])
      j++;
    else {
      double w = text_weight(a[i], country, type);
      ab += w * w;
      shared += w;
      i++;
      j++;
    }
  }
  return {float(aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : 0),
          float(wa > 0 ? shared / wa : 0), float(wb > 0 ? shared / wb : 0)};
}
#endif
string support_name(const string &n) {
  string name = join(toks(alias_name(n)));
  auto it = aliases.find("%" + name);
  return it == aliases.end() ? name : it->second;
}
SupportRec support_parse(const string &line) {
  auto x = split(line, '\t');
  string name = support_name(x[2]),
         address = join(toks(alias_name(x[3], true)));
  SupportRec r{name, compact(name), address, nums(toks(address)),
               grams(address)};
#ifdef COPY_SUPPORT
  if (!name.empty() && !address.empty())
    r.signature = hashkey(name + "\t" + address);
#endif
  return r;
}
struct SecondRow {
  U ref, target;
  array<float, SECOND_NF> f;
};
#ifdef COPY_SUPPORT
struct CopyEvidence {
  U source2 = 0, source3 = 0;
  float sum = 0;
};
void append_copy_features(const Mapping<Row> &rows, const Mapping<float> &prob,
                          char **argv) {
  struct PreviousRow {
    U ref, target;
    array<float, NF + 32> f;
  };
  Mapping<PreviousRow> previous(argv[12]);
  if (previous.n != rows.n)
    throw runtime_error("append feature alignment");
  size_t nr = 0;
  string line;
  ifstream refs(argv[5]);
  while (getline(refs, line))
    nr++;
  vector<vector<U>> byref(nr);
  unordered_set<U> wanted;
  for (size_t j = 0; j < rows.n; j++) {
    if (rows[j].ref >= nr || previous[j].ref != rows[j].ref ||
        previous[j].target != rows[j].target || previous[j].f[NF] != prob[j])
      throw runtime_error("append pair alignment");
    byref[rows[j].ref].push_back(j);
    wanted.insert(rows[j].target);
  }
  unordered_map<U, H> signatures;
  signatures.reserve(wanted.size());
  for (int source = 3; source <= 4; source++) {
    ifstream input(argv[source]);
    while (getline(input, line)) {
      auto x = split(line, '\t');
      U target = stoul(x[0].substr(3)) | (x[0][1] == '3' ? 0x80000000U : 0U);
      if (!wanted.count(target))
        continue;
      string name = support_name(x[2]);
      string address = join(toks(alias_name(x[3], true)));
      signatures.emplace(target, name.empty() || address.empty()
                                     ? 0
                                     : hashkey(name + "\t" + address));
    }
  }
  if (signatures.size() != wanted.size())
    throw runtime_error("append missing record");
  wanted.clear();
  wanted.rehash(0);
  vector<CopyEvidence> evidence(rows.n);
  for (auto &indices : byref) {
    unordered_map<H, CopyEvidence> counts;
    for (U j : indices) {
      H signature = signatures.at(rows[j].target);
      if (!signature)
        continue;
      auto &count = counts[signature];
      (rows[j].target >> 31 ? count.source3 : count.source2)++;
      count.sum += prob[j];
    }
    for (U j : indices) {
      H signature = signatures.at(rows[j].target);
      if (signature)
        evidence[j] = counts.at(signature);
    }
  }
  ofstream output(argv[6], ios::binary);
  for (size_t j = 0; j < rows.n; j++) {
    SecondRow row{};
    row.ref = rows[j].ref;
    row.target = rows[j].target;
    copy(previous[j].f.begin(), previous[j].f.end(), row.f.begin());
    auto &count = evidence[j];
    U peers = count.source2 + count.source3;
    if (peers > 1) {
      row.f[NF + 32] = peers - 1;
      row.f[NF + 33] = count.source2 - ((row.target >> 31) == 0);
      row.f[NF + 34] = count.source3 - ((row.target >> 31) == 1);
      row.f[NF + 35] = (count.sum - prob[j]) / (peers - 1);
    }
    output.write((const char *)&row, sizeof(row));
  }
  cerr << "Appended copy features " << rows.n << endl;
}
#endif
static_assert(sizeof(SecondRow) == (SECOND_NF + 2) * sizeof(float));
int main(int argc, char **argv) {
  try {
    if (argc < 12)
      throw runtime_error("collective FEATURES PROBABILITIES TARGET2 TARGET3 "
                          "REFS OUT ALIASES train|score MODEL LIBRARY SOURCE1");
    Mapping<Row> rows(argv[1]);
    Mapping<float> prob(argv[2]);
    if (rows.n != prob.n)
      throw runtime_error("probability alignment");
    ifstream af(argv[7]);
    string line;
    while (getline(af, line)) {
      auto x = split(line, '\t');
      if (x.size() == 2)
        aliases[x[0]] = x[1];
    }
#ifdef COPY_SUPPORT
    if (string(argv[8]) == "append") {
      if (argc < 13)
        throw runtime_error("append requires existing 166-column features");
      append_copy_features(rows, prob, argv);
      return 0;
    }
#endif
    size_t nr = 0;
    unordered_map<U, U> refmap;
    ifstream rf(argv[5]);
    while (getline(rf, line)) {
      refmap[stoul(line.substr(3, line.find('\t') - 3))] = nr++;
    }
    string mode = argv[8];
    bool ranged = mode == "score_range" || mode == "train_range";
    if (ranged && argc < 14)
      throw runtime_error("range mode requires reference start and stop");
    size_t ref_start = ranged ? stoull(argv[12]) : 0;
    size_t ref_stop = ranged ? stoull(argv[13]) : nr;
    if (ref_start > ref_stop || ref_stop > nr)
      throw runtime_error("invalid reference range");
    vector<unsigned char> has_rows(nr, 0);
    for (size_t j = 0; j < rows.n; j++) {
      if (rows[j].ref >= nr)
        throw runtime_error("reference index");
      has_rows[rows[j].ref] = 1;
    }
    auto active = [&](size_t i) {
      return ref_start <= i && i < ref_stop && has_rows[i];
    };
    vector<string> refnames(nr);
#ifdef EXTRA_TFIDF
    vector<TextRec> reftext(nr);
#endif
    ifstream rn(argv[11]);
    while (getline(rn, line)) {
      auto x = split(line, '\t');
      auto it = refmap.find(stoul(x[0].substr(3)));
      if (it != refmap.end()) {
        if (active(it->second))
          refnames[it->second] = support_name(x[2]);
#ifdef EXTRA_TFIDF
        auto t = text_parse(line);
        country_size[t.country]++;
        for (int type = 0; type < 4; type++) {
          auto &v = type == 0   ? t.ng
                    : type == 1 ? t.ag
                    : type == 2 ? t.nw
                                : t.aw;
          for (H g : v)
            text_df[text_key(g, t.country, type)]++;
        }
        if (active(it->second))
          reftext[it->second] = move(t);
#endif
      }
    }
    refmap.clear();
    refmap.rehash(0);
    vector<vector<U>> byref(nr);
    unordered_set<U> wanted;
    wanted.reserve(min(rows.n, (ref_stop - ref_start) * 16));
    vector<U> scoring_rows;
    for (size_t j = 0; j < rows.n; j++) {
      if (rows[j].ref >= nr)
        throw runtime_error("reference index");
      if (!active(rows[j].ref))
        continue;
      scoring_rows.push_back(j);
      byref[rows[j].ref].push_back(j);
      wanted.insert(rows[j].target);
    }
    unordered_map<U, SupportRec> records;
    records.reserve(wanted.size());
#ifdef EXTRA_TFIDF
    unordered_map<U, TextRec> targettext;
    targettext.reserve(wanted.size());
#endif
    for (int s = 3; s <= 4; s++) {
      ifstream inp(argv[s]);
      while (getline(inp, line)) {
        auto p = line.find('\t');
        string id = line.substr(0, p);
        U t = stoul(id.substr(3)) | (id[1] == '3' ? 0x80000000U : 0U);
        if (wanted.count(t)) {
          records.emplace(t, support_parse(line));
#ifdef EXTRA_TFIDF
          auto text = text_parse(line);
          vector<H>().swap(text.ag);
          targettext.emplace(t, move(text));
#endif
        }
      }
      cerr << "support records " << records.size() << endl;
    }
    if (records.size() != wanted.size())
      throw runtime_error("missing support record");
    wanted.clear();
    wanted.rehash(0);
#ifdef COPY_SUPPORT
    vector<CopyEvidence> copy_evidence(rows.n);
    for (auto &v : byref) {
      unordered_map<H, CopyEvidence> counts;
      for (U j : v) {
        H signature = records.at(rows[j].target).signature;
        if (!signature)
          continue;
        auto &c = counts[signature];
        (rows[j].target >> 31 ? c.source3 : c.source2)++;
        c.sum += prob[j];
      }
      for (U j : v) {
        H signature = records.at(rows[j].target).signature;
        if (signature)
          copy_evidence[j] = counts.at(signature);
      }
    }
#endif
    vector<vector<U>> support(nr);
    for (size_t i = 0; i < nr; i++) {
      auto &v = support[i];
      for (U j : byref[i])
        if (prob[j] >= .8f)
          v.push_back(j);
      sort(v.begin(), v.end(), [&](U a, U b) { return prob[a] > prob[b]; });
      if (v.size() > 6)
        v.resize(6);
    }
    bool training = mode == "train" || mode == "train_range";
    unique_ptr<LightGBM> model;
    if (!training) {
      if (argc < 11)
        throw runtime_error("library missing");
      model = make_unique<LightGBM>(argv[10], argv[9]);
    }
    ofstream out(argv[6], ios::binary);
    uint64_t nextlog = 500000;
    for (size_t start = 0; start < scoring_rows.size(); start += 50000) {
      size_t count = min(size_t(50000), scoring_rows.size() - start);
      vector<SecondRow> batch(count);
#pragma omp parallel for num_threads(4) schedule(dynamic, 64)
      for (size_t local = 0; local < count; local++) {
        size_t j = scoring_rows[start + local];
        const Row &r = rows[j];
        auto &b = batch[local];
        b.ref = r.ref;
        b.target = r.target;
        copy(r.f.begin(), r.f.end(), b.f.begin());
        auto &q = records.at(r.target);
        array<float, SUPPORT_NF> f{};
        f[0] = prob[j];
        float sum = 0;
        int n = 0;
        for (U k : byref[r.ref])
          if (rows[k].target != r.target) {
            float p = prob[k];
            sum += p;
            n++;
            if (p > f[1]) {
              f[2] = f[1];
              f[1] = p;
            } else
              f[2] = max(f[2], p);
            f[4] += p >= .5f;
            f[5] += p >= .8f;
            f[6] += p >= .95f;
            f[7] += p >= .99f;
            if (p >= .8f)
              f[rows[k].target >> 31 ? 9 : 8]++;
          }
        f[3] = n ? sum / n : 0;
        int used = 0;
        for (U k : support[r.ref]) {
          if (rows[k].target == r.target || used++ >= 5)
            continue;
          auto &t = records.at(rows[k].target);
          float ns = lcs(q.name, t.name),
                nc = lcs(q.compactname, t.compactname), ad = gdice(q.ag, t.ag),
                as = lcs(q.address, t.address);
          int ni = intersection(q.numbers, t.numbers);
          f[10] = max(f[10], ns);
          f[11] = max(f[11], nc);
          f[12] = max(f[12], ad);
          f[13] = max(f[13], as);
          f[14] = max(f[14], max(ns, nc) * (.42f + .58f * max(ad, as)));
          f[15] =
              max(f[15], frac(ni, q.numbers.size() + t.numbers.size() - ni));
          f[16] = max(f[16], frac(ni, min(q.numbers.size(), t.numbers.size())));
          f[17] =
              max(f[17], float(q.numbers == t.numbers && !q.numbers.empty()));
          f[18] += q.name == t.name && !q.name.empty();
          f[19] += q.address == t.address && !q.address.empty();
          f[20] += t.numbers.empty();
          f[21] = max(f[21], max(ns, nc) * prob[k]);
          f[22] = max(f[22], max(ad, as) * prob[k]);
        }
        float other = 0;
        for (size_t k = j; k > 0 && rows[k - 1].target == r.target;) {
          --k;
          other = max(other, prob[k]);
        }
        for (size_t k = j + 1; k < rows.n && rows[k].target == r.target; k++)
          other = max(other, prob[k]);
        f[23] = other;
        f[24] = prob[j] - other;
        f[25] = lcs(refnames[r.ref], q.name);
#ifdef EXTRA_TFIDF
        auto &a = reftext[r.ref];
        auto &t = targettext.at(r.target);
        if (a.country != t.country)
          throw runtime_error("TFIDF country mismatch");
        auto name_similarity = text_similarity(a.ng, t.ng, a.country, 0);
        f[26] = name_similarity[0];
        f[27] = text_similarity(a.ag, q.ag, a.country, 1)[0];
        f[28] = text_similarity(a.nw, t.nw, a.country, 2)[0];
        f[29] = text_similarity(a.aw, t.aw, a.country, 3)[0];
        f[30] = name_similarity[1];
        f[31] = name_similarity[2];
#endif
#ifdef COPY_SUPPORT
        auto &copies = copy_evidence[j];
        U peers = copies.source2 + copies.source3;
        if (peers > 1) {
          f[32] = peers - 1;
          f[33] = copies.source2 - ((r.target >> 31) == 0);
          f[34] = copies.source3 - ((r.target >> 31) == 1);
          f[35] = (copies.sum - prob[j]) / (peers - 1);
        }
#endif
        copy(f.begin(), f.end(), b.f.begin() + NF);
      }
      if (training)
        out.write((const char *)batch.data(), batch.size() * sizeof(SecondRow));
      else {
        vector<float> x;
        x.reserve(count * SECOND_NF);
        for (auto &b : batch)
          x.insert(x.end(), b.f.begin(), b.f.end());
        auto p = model->score_matrix(x, SECOND_NF);
        for (size_t j = 0; j < count; j++) {
          Scored r{batch[j].ref, batch[j].target, float(p[j])};
          out.write((const char *)&r, sizeof(r));
        }
      }
      if (start + count >= nextlog) {
        cerr << "support pairs " << start + count << endl;
        nextlog += 500000;
      }
    }
  } catch (const exception &e) {
    cerr << e.what() << endl;
    return 2;
  }
  return 0;
}
