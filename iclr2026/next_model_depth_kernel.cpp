// Exact threshold-chain DFS with optimistic per-query repair-cost bounds.
#include <algorithm>
#include <vector>
#include <cstdint>
#include <chrono>
#include <limits>
extern "C" int search_depth(int n,int k,const int* ranks,const int* y,const double* costs,
    double limit,double seconds_limit,const double* initial,double* best,int* witness,uint64_t* stats) {
    std::vector<double> bound(initial,initial+n+1);
    std::vector<int> path(k-1,0), active(n);
    for(int i=0;i<n;++i) active[i]=i;
    std::vector<std::vector<int>> sorted(k-1,active);
    std::vector<std::vector<char>> member(k-1,std::vector<char>(n,0));
    for(int s=0;s<k-1;++s) std::stable_sort(sorted[s].begin(),sorted[s].end(),[&](int a,int b){return ranks[s*n+a]<ranks[s*n+b];});
    double c0=0;int q0=0;
    for(int i=0;i<n;++i){c0+=costs[i];q0+=y[i];}
    auto start=std::chrono::steady_clock::now();
    bool complete=true;
    double eps=1e-12*n;
    auto save=[&](int stage,double c,int q){
        if(c<best[q]){
            best[q]=c;
            for(int j=0;j<k-1;++j) witness[(k-1)*q+j]=j<stage?path[j]:0;
        }
        if(c<bound[q]) for(int j=q;j>=0 && c<bound[j];--j) bound[j]=c;
    };
    auto dfs = [&](auto&& self,int stage,const std::vector<int>& reach,double c,int q)->void {
        if(!complete || c>limit) return;
        ++stats[0];
        if(seconds_limit>0 && stats[0]%1024==0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>seconds_limit){complete=false;return;}
        save(stage,c,q);
        if(stage==k-1 || reach.empty()) return;
        // An optimistic continuation repairs current errors independently, with
        // no harmful switches and no threshold constraints. Each repair must
        // pay every intervening generation before its first correct answer.
        std::vector<double> repairs;
        for(int i:reach) if(y[stage*n+i]==0){
            double charge=0;
            for(int j=stage+1;j<k;++j){
                charge+=costs[j*n+i];
                if(y[j*n+i]){repairs.push_back(charge);break;}
            }
        }
        if(bound[q+repairs.size()]<c-eps){++stats[1];return;}
        std::sort(repairs.begin(),repairs.end());
        double lower=c;bool possible=false;
        for(int j=0;j<int(repairs.size());++j){
            lower+=repairs[j];
            if(lower>limit+eps) break;
            if(lower<=bound[q+j+1]+eps){possible=true;break;}
        }
        if(!possible){++stats[1];return;}
        std::vector<int> order, next;
        order.reserve(reach.size());
        for(int i:reach)member[stage][i]=1;
        for(int i:sorted[stage])if(member[stage][i])order.push_back(i);
        for(int i:reach)member[stage][i]=0;
        next.reserve(order.size());
        int pos=0,maxq=q;
        while(pos<int(order.size())){
            int r=ranks[stage*n+order[pos]];
            do {int i=order[pos++];next.push_back(i);c+=costs[(stage+1)*n+i];q+=y[(stage+1)*n+i]-y[stage*n+i];}
            while(pos<int(order.size()) && ranks[stage*n+order[pos]]==r);
            if(c>limit) break;
            path[stage]=r+1;
            if(stage==k-2){
                if(q>maxq){save(stage+1,c,q);maxq=q;}
            }else self(self,stage+1,next,c,q);
        }
        path[stage]=0;
    };
    dfs(dfs,0,active,c0,q0);
    return complete?1:0;
}
