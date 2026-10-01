// Exact empirical four-stage search. Costs are sums, correctness is integer.
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

extern "C" uint64_t search_four(int n, const int* ranks, const int* quality,
                                const double* cost, double limit,
                                double* best, int* witness) {
    const int *ra=ranks, *rb=ranks+n, *rc=ranks+2*n;
    std::vector<int> oa(n), ob(n);
    for (int i=0;i<n;++i) oa[i]=ob[i]=i;
    std::stable_sort(oa.begin(),oa.end(),[&](int i,int j){return ra[i]<ra[j];});
    std::stable_sort(ob.begin(),ob.end(),[&](int i,int j){return rb[i]<rb[j];});
    int nc=*std::max_element(rc,rc+n)+1;
    std::vector<double> hc(nc);
    std::vector<int> hq(nc), hits(nc);
    double basec=0.; int baseq=0;
    for(int i=0;i<n;++i){basec+=cost[i];baseq+=quality[i];}
    uint64_t evaluated=0;
    auto save=[&](double c,int q,int a,int b,int z){
        ++evaluated;
        if(c<=limit && c<best[q]){
            best[q]=c; witness[3*q]=a; witness[3*q+1]=b; witness[3*q+2]=z;
        }
    };
    int ap=0, a=0;
    while(true){
        if(basec>limit) break;
        std::fill(hc.begin(),hc.end(),0.);
        std::fill(hq.begin(),hq.end(),0);
        std::fill(hits.begin(),hits.end(),0);
        double bc=basec; int bq=baseq;
        save(bc,bq,a,0,0);
        int bp=0;
        while(bp<n){
            int group=rb[ob[bp]]; bool changed=false;
            while(bp<n && rb[ob[bp]]==group){
                int i=ob[bp++];
                if(ra[i]>=a) continue;
                changed=true;
                bc+=cost[2*n+i]; bq+=quality[2*n+i]-quality[n+i];
                hc[rc[i]]+=cost[3*n+i];
                hq[rc[i]]+=quality[3*n+i]-quality[2*n+i];
                ++hits[rc[i]];
            }
            if(!changed) continue;
            if(bc>limit) break;
            double cc=bc; int cq=bq;
            save(cc,cq,a,group+1,0);
            for(int z=0;z<nc;++z){
                if(!hits[z]) continue;
                cc+=hc[z]; cq+=hq[z];
                if(cc>limit) break;
                save(cc,cq,a,group+1,z+1);
            }
        }
        if(ap==n) break;
        int group=ra[oa[ap]];
        while(ap<n && ra[oa[ap]]==group){
            int i=oa[ap++];
            basec+=cost[n+i]; baseq+=quality[n+i]-quality[i];
        }
        a=group+1;
    }
    return evaluated;
}
