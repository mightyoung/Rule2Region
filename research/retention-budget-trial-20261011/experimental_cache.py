"""Explicit opt-in experiment; no production defaults or motion gates changed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cache-diagnostics-dev-20261011'))
from development_cache import DevelopmentCache
from vendor.base_cache import clock


class ExperimentalCache(DevelopmentCache):
    def protected_endpoints(self,scope,quota):
        protected=set()
        for _,r in sorted(self.rels.items(),key=lambda x:(x[1]['_seq'],x[0])):
            if r['status']!='accepted_hypothesis':continue
            pair={r['a'],r['b']}
            if not pair<=scope:continue
            key=r['a'][:2];now=self.keys[key]['now']
            if self._active.get(key[0])!=key[1]:continue
            if clock(r['available_at'])>now:continue
            if any(self.obs[i]['_t']<now-self.ttl for i in pair):continue
            if r['endpoint_bindings']!=[self.binding(self.obs[i]) for i in (r['a'],r['b'])]:continue
            if len(protected|pair)<=quota:protected|=pair
        return protected

    def _victim(self,scope,limit,order):
        protected=self.protected_endpoints(set(scope),limit//4)
        eligible=[i for i in scope if i not in protected]
        # Defensive fallback preserves termination if a future caller breaks
        # the quota contract; normal configured quotas always leave candidates.
        return min(eligible or list(scope),key=order)

    def bounds(self,key):
        while len([i for i in self.obs if i[:2]==key])>self.po:
            scope=[i for i in self.obs if i[:2]==key]
            victim=self._victim(scope,self.po,lambda i:(self.obs[i]['_t'],i))
            self.remove_obs(victim,'evicted_observation')
        while len(self.obs)>self.to:
            victim=self._victim(list(self.obs),self.to,lambda i:(self.obs[i]['_seq'],i))
            self.remove_obs(victim,'evicted_observation')
        # Existing relation bounds, cascades, stats and callbacks stay intact.
        super().bounds(key)
