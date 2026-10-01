#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

struct State { std::vector<int> t; int q=-1; double c=INFINITY; };
struct Slice { std::vector<double> c, ratio; std::vector<int> q; };
struct Search {
 int n,d,w,iters; const int *rank,*groups,*quality; const double *cost;
 std::vector<std::vector<int>> order;
 uint64_t slices=0, proposals=0, iterations=0, starts=0, transfer_accepts=0;
 bool better(const State&a,const State&b) {return a.q>b.q || (a.q==b.q && a.c<b.c-1e-15);}
 State evaluate(std::vector<int> t) {
  State s{t,0,0.};
  for(int x=0;x<n;x++) for(int j=0;j<d;j++) {
   s.c+=cost[j*n+x];
   if(j==d-1 || rank[j*n+x]>=t[j]) {s.q+=quality[j*n+x];break;}
  }
  s.c/=n;return s;
 }
 Slice slice(const std::vector<int>&t,int i) {
  slices++;
  int g=groups[i];
  std::vector<double> dc(g,0.); std::vector<int> dq(g,0),cnt(g,0);
  double basec=0.;int baseq=0;
  // Individual reached-query increments are sorted by stage score rank.
  std::vector<std::pair<int,std::pair<int,double>>> reached;reached.reserve(n);
  for(int x:order[i]) {
   bool reach=true;
   for(int j=0;j<i;j++) {
    basec+=cost[j*n+x];
    if(rank[j*n+x]>=t[j]) {baseq+=quality[j*n+x];reach=false;break;}
   }
   if(!reach)continue;
   basec+=cost[i*n+x];baseq+=quality[i*n+x];
   double suffixc=0.;int suffixq=0;
   for(int j=i+1;j<d;j++) {
    suffixc+=cost[j*n+x];
    if(j==d-1 || rank[j*n+x]>=t[j]) {suffixq=quality[j*n+x];break;}
   }
   int r=rank[i*n+x], gain=suffixq-quality[i*n+x];
   dc[r]+=suffixc;dq[r]+=gain;cnt[r]++;
   reached.push_back({r,{gain,suffixc}});
  }
  Slice s;s.c.resize(g+1);s.q.resize(g+1);s.ratio.assign(g+1,NAN);
  s.c[0]=basec/n;s.q[0]=baseq;
  for(int r=0;r<g;r++) {basec+=dc[r];baseq+=dq[r];s.c[r+1]=basec/n;s.q[r+1]=baseq;}
  // A fixed nearest-rank window, with one-sided estimates at endpoints.
  if(reached.size()>=50) {
   int nr=reached.size(),width=std::min(w,nr),pos=0;
   std::vector<double> cg(nr+1,0.),cc(nr+1,0.);
   for(int a=0;a<nr;a++){cg[a+1]=cg[a]+reached[a].second.first;cc[a+1]=cc[a]+reached[a].second.second;}
   for(int r=0;r<=g;r++) {
    int lo=std::max(0,std::min(nr-width,pos-width/2)),hi=lo+width;
    double den=cc[hi]-cc[lo];
    if(den>0)s.ratio[r]=(cg[hi]-cg[lo])/den;
    if(r<g)pos+=cnt[r];
   }
  }
  return s;
 }
 State coordinate(const State&initial,int i,double budget) {
  auto s=slice(initial.t,i);
  State best=initial;
  if(best.c>budget+1e-12){best.q=-1;best.c=INFINITY;}
  auto consider=[&](int r){
   if(r<0 || r>groups[i] || s.c[r]>budget+1e-12)return;
   proposals++;
   State a{initial.t,s.q[r],s.c[r]};a.t[i]=r;
   if(better(a,best))best=a;
  };
  consider(0);consider(groups[i]);
  int upper=std::upper_bound(s.c.begin(),s.c.end(),budget+1e-12)-s.c.begin()-1;
  consider(upper);
  for(int r=0;r<groups[i];r++) {
   double a=s.ratio[r],b=s.ratio[r+1];
   if((a>0 && b<=0)||(a>=0 && b<0)){consider(r);consider(r+1);}
  }
  return best;
 }
 State optimize(State s,double budget) {
  starts++;
  // Repair an infeasible start through its first threshold. With no first
  // escalation, its cost equals the first standalone model's cost.
  s=coordinate(s,0,budget);
  for(int it=0;it<iters;it++) {
   iterations++;
   State before=s;
   for(int i=0;i<d-1;i++)s=coordinate(s,i,budget);
   std::vector<double> ratio(d-1);
   for(int i=0;i<d-1;i++){auto z=slice(s.t,i);ratio[i]=z.ratio[s.t[i]];}
   State transfer=s;
   for(int donor=0;donor<d-1;donor++) for(int recv=0;recv<d-1;recv++) {
    if(donor==recv || !std::isfinite(ratio[donor]) || !std::isfinite(ratio[recv]) || ratio[recv]<=std::max(0.,ratio[donor]) || s.t[donor]==0 || s.t[recv]==groups[recv])continue;
    for(double fraction: {0.01,0.05,0.20}) {
     auto t=s.t;t[donor]=std::max(0,t[donor]-std::max(1,(int)std::ceil(fraction*groups[donor])));
     State a=evaluate(t);a=coordinate(a,recv,budget);
     if(a.t[recv]>s.t[recv] && better(a,transfer))transfer=a;
    }
   }
   if(better(transfer,s)){s=transfer;transfer_accepts++;}
   if(!better(s,before))break;
  }
  return s;
 }
};

extern "C" void search_foc_chain(int n,int d,const int*rank,const int*groups,const int*quality,const double*cost,int nb,const double*budgets,int window,int iters,int stride,int*out_t,double*out_c,int*out_q,uint64_t*stats) {
 Search search{n,d,window,iters,rank,groups,quality,cost};
 search.order.resize(d-1);
 for(int i=0;i<d-1;i++){auto &o=search.order[i];o.resize(n);std::iota(o.begin(),o.end(),0);std::stable_sort(o.begin(),o.end(),[&](int a,int b){return rank[i*n+a]<rank[i*n+b];});}
 State warm;warm.t.assign(d-1,0);
 double first=0.;for(int x=0;x<n;x++)first+=cost[x]/n;
 bool initialized=false;
 for(int b=0;b<nb;b++) {
  if(first>budgets[b]+1e-12){out_c[b]=INFINITY;out_q[b]=-1;for(int i=0;i<d-1;i++)out_t[b*(d-1)+i]=0;continue;}
  State best=search.optimize(search.evaluate(warm.t),budgets[b]);
  if(!initialized || b%stride==0) {
   for(double level:{.25,.50,.75,1.}) {
    std::vector<int>t(d-1);for(int i=0;i<d-1;i++)t[i]=(int)std::round(level*groups[i]);
    State a=search.optimize(search.evaluate(t),budgets[b]);
    if(search.better(a,best))best=a;
   }
  }
  initialized=true;warm=best;
  out_c[b]=best.c;out_q[b]=best.q;
  for(int i=0;i<d-1;i++)out_t[b*(d-1)+i]=best.t[i];
 }
 stats[0]=search.slices;stats[1]=search.proposals;stats[2]=search.iterations;stats[3]=search.starts;stats[4]=search.transfer_accepts;
}
