// Stream four tied-score thresholds. No dense four-dimensional grid.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

extern "C" int search_five(int n, const int* ranks, const int* quality,
                           const double* cost, double limit, double seconds_limit,
                           double* best, int* witness, uint64_t* stats) {
    const int *ra=ranks, *rb=ranks+n, *rc=ranks+2*n, *rd=ranks+3*n;
    std::vector<int> oa(n), ob(n), oc(n);
    for(int i=0;i<n;++i) oa[i]=ob[i]=oc[i]=i;
    std::stable_sort(oa.begin(),oa.end(),[&](int i,int j){return ra[i]<ra[j];});
    std::stable_sort(ob.begin(),ob.end(),[&](int i,int j){return rb[i]<rb[j];});
    std::stable_sort(oc.begin(),oc.end(),[&](int i,int j){return rc[i]<rc[j];});
    int nd=*std::max_element(rd,rd+n)+1;
    std::vector<double> hc(nd);
    std::vector<int> hq(nd);
    std::vector<uint64_t> mask((nd+63)/64);
    double basec=0.; int baseq=0;
    for(int i=0;i<n;++i){basec+=cost[i];baseq+=quality[i];}
    const auto start=std::chrono::steady_clock::now();
    auto expired=[&](){return seconds_limit>0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>seconds_limit;};
    auto save=[&](double c,int q,int a,int b,int z,int d){
        ++stats[0];
        if(c<=limit && c<best[q]){
            best[q]=c; witness[4*q]=a; witness[4*q+1]=b;
            witness[4*q+2]=z; witness[4*q+3]=d;
        }
    };
    int ap=0, a=0;
    while(true){
        if(basec>limit) break;
        double bc=basec; int bq=baseq;
        save(bc,bq,a,0,0,0);
        int bp=0;
        while(bp<n){
            if(expired()) return 0;
            int bg=rb[ob[bp]]; bool changed=false;
            while(bp<n && rb[ob[bp]]==bg){
                int i=ob[bp++];
                if(ra[i]>=a) continue;
                changed=true;
                bc+=cost[2*n+i]; bq+=quality[2*n+i]-quality[n+i];
            }
            if(!changed) continue;
            if(bc>limit) break;
            std::fill(hc.begin(),hc.end(),0.);
            std::fill(hq.begin(),hq.end(),0);
            std::fill(mask.begin(),mask.end(),0);
            double cc=bc; int cq=bq;
            save(cc,cq,a,bg+1,0,0);
            int cp=0;
            while(cp<n){
                int cg=rc[oc[cp]]; bool cchanged=false;
                while(cp<n && rc[oc[cp]]==cg){
                    int i=oc[cp++];
                    if(ra[i]>=a || rb[i]>bg) continue;
                    cchanged=true;
                    cc+=cost[3*n+i]; cq+=quality[3*n+i]-quality[2*n+i];
                    hc[rd[i]]+=cost[4*n+i]; hq[rd[i]]+=quality[4*n+i]-quality[3*n+i];
                    mask[rd[i]/64] |= uint64_t(1)<<(rd[i]%64);
                }
                if(!cchanged) continue;
                if(cc>limit) break;
                double dc=cc; int dq=cq;
                save(dc,dq,a,bg+1,cg+1,0);
                bool over=false;
                for(int word=0;word<int(mask.size()) && !over;++word){
                    uint64_t bits=mask[word];
                    while(bits){
                        int d=64*word+__builtin_ctzll(bits);
                        bits &= bits-1;
                        dc+=hc[d]; dq+=hq[d];
                        if(dc>limit){over=true;break;}
                        save(dc,dq,a,bg+1,cg+1,d+1);
                    }
                }
            }
        }
        ++stats[1];
        if(ap==n) break;
        int ag=ra[oa[ap]];
        while(ap<n && ra[oa[ap]]==ag){
            int i=oa[ap++];
            basec+=cost[n+i]; baseq+=quality[n+i]-quality[i];
        }
        a=ag+1;
    }
    return 1;
}
