"""Incremental rich report. Original report and all original artifacts stay unchanged."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from reproduce_v1 import v1
ROOT=Path(__file__).resolve().parent

def main():
    revised=json.loads((ROOT/'results_v2.json').read_text())
    data={n:v1.read(n+'.json') for n in ['results','topology','goals','states_gt','events_gt','observations','admission']}
    data['results']['protocol_sha256']=hashlib.sha256((ROOT/'PROTOCOL_V2.md').read_bytes()).hexdigest()
    for name,r in revised['strategies'].items():
        old=data['results']['strategies'][name]
        replacement=deepcopy(r)
        if 'config' in old: replacement['config']=old['config']
        replacement['reasons']=r.get('config_reasons',[])+r.get('session_reasons',[])
        # Rejected v2 arms receive no forced diagnostic prediction. Original retained separately.
        data['results']['strategies'][name]=replacement
    template=(v1.ROOT/'report_template.html').read_text()
    template=template.replace('整栋边界配置小试','准入修正 v2 · 整栋边界配置小试')
    template=template.replace('整栋目标，如何落到每一扇门？','准入修正 v2：合法数值还不够')
    template=template.replace('INDEPENDENT PILOT /','INCREMENTAL ADMISSION V2 /')
    template=template.replace('<label><input type="checkbox" id="diagnostic">显示被拒配置的强制诊断</label>', '<input type="checkbox" id="diagnostic" hidden disabled><span class="small muted">v2 拒绝后不计算；旧反例在上方保留</span>')
    template=template.replace("r.status==='SYNTHETIC_ADMITTED'","r.status==='SYNTHETIC_V2_ADMITTED'")
    # Relative links remain to preserved original docs, except new corrected report evidence.
    for name in ['OFFICIAL_ADMISSION.md','real_gt_ledger.template.json','BASELINE_CHECK.md']:
        template=template.replace(f'href="{name}"',f'href="../building_boundary/{name}"')
    template=template.replace('href="PROTOCOL.md"','href="PROTOCOL_V2.md"').replace('href="results.json"','href="results_v2.json"')
    template=template.replace('原冻结 16 图协议、旧工作树、main 与原 README 均未改动。','v1 全部文件与原报告保持不变；旧有限夹具零误差保留，不代表旧准入对任意候选可靠。')
    template=template.replace('审批收据仅为合成夹具模拟。','审批收据仅为合成夹具模拟。v2 配置审批和初值会话资格分开绑定；场景侧证据是自制坐标参考，未审核真实现场。')
    names={'reverse_line_only':'只反转有向线','displaced_valid_roi':'合法但错位 ROI','changed_initial_old_approval':'初值 30/20/10，保留旧审批'}
    rows=''
    for name,label in names.items():
        before=revised['counterexamples_v1'][name];after=revised['failures_v2'][name];m=before['metrics']['building']
        reason='; '.join(after['config_reasons']+after['session_reasons'])
        rows+=f'<tr><td>{label}</td><td>旧版准入 · max {m["max_abs_error"]}<br>FP / FN {m["event_fp"]} / {m["event_fn"]}</td><td>{after["config_status"]}</td><td>{after["session_status"]}</td><td><code>{reason}</code><br>无预测 / metrics=null</td></tr>'
    positive=''.join(f'<li>{name}：{run["status"]}，六目标最大误差均 0。</li>' for name,run in revised['non_oracle_positive_controls'].items())
    failures=''.join(f'<tr><td>{name}</td><td>{run["config_status"]}</td><td>{run["session_status"]}</td><td><code>{"; ".join(run["config_reasons"]+run["session_reasons"])}</code></td></tr>' for name,run in revised['failures_v2'].items())
    panel=f'''<section class="panel"><h2>先保留失败，再修正准入</h2><p><a href="../building_boundary/report.html">原 v1 报告（未改动）</a> · <a href="v1_counterexamples.json">完整旧版反例</a> · <a href="PROTOCOL_V2.md">修正与下一阶段协议</a></p><p>三项反例已复现。原软件通过不等于任何候选可靠，零误差只覆盖旧固定夹具。v2 的被拒运行不给预测或正式成绩。</p><div class="table-wrap"><table><thead><tr><th>反例</th><th>旧版整栋误差</th><th>v2 配置审批层</th><th>v2 初值会话层</th><th>拒绝证据</th></tr></thead><tbody>{rows}</tbody></table></div><h3>分层检查，保留未知</h3><ol><li>归一化数值合法，不证明几何或现场正确。</li><li>有限线与 ROI 有正长度交集，不证明线位于门口。</li><li>经审核参考资料提供通道支持区域及 a/b 侧锚点；方向相对于候选有向线检验，不对照计数 GT。</li><li>配置审批绑定目标、拓扑、候选和场景证据；初值会话资格另绑定 t0、来源、版本和配置审批。</li></ol><p class="muted">这里的参考和审批均为合成作者模拟，没有视觉识门、现场几何审核或真实人员授权。软件无法识别伪造但自洽的盘点记录，不能代替现场核验。</p><h3>不把几何固定成答案的正对照</h3><ul>{positive}</ul><details><summary>全部 12 项修正拒绝记录</summary><div class="table-wrap"><table><thead><tr><th>情形</th><th>配置层</th><th>会话层</th><th>原因</th></tr></thead><tbody>{failures}</tbody></table></div></details></section>'''
    marker='<div class="controls">'
    template=template.replace(marker,panel+marker)
    template=template.replace('<h2>失败与未知保持可见</h2>','<h2>保留的 v1 原失败检查</h2>')
    template=template.replace('以下均为实际软件闸门测试。','以下为原 v1 已运行的软件闸门测试，原结果原样保留；新增修正反例见上方 v2 表。')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (ROOT/'report_v2.html').write_text(template.replace('__DATA__',payload))
    print('Incremental report_v2.html generated; no original file changed')

if __name__=='__main__':main()
