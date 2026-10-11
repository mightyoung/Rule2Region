"""Focused, self-contained incremental report. Earlier reports are read-only history."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    r=json.loads((ROOT/'results_v3.json').read_text())
    rows=''
    labels={'short_finite_line':'Important · 短有限线','missing_source_id_and_revision':'Minor · 来源字段双方缺失'}
    for name,case in r['counterexamples'].items():
        before,after=case['v2_false_admission'],case['v3'];m=before['metrics']['building']
        rows+=f'<tr><td>{labels[name]}</td><td>{before["status"]}<br>max {m["max_abs_error"]} / FN {m["event_fn"]}</td><td>{after["status"]}<br><code>{"; ".join(after["config_reasons"]+after["session_reasons"])}</code><br>无预测；metrics=null</td></tr>'
    payload=json.dumps(r,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rule2Region · 有限计数线补修 v3</title>
<style>body{margin:0;background:#f3f6fa;color:#192b40;font:15px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 20px}h1{font-size:30px}h2{font-size:20px;margin:0 0 12px}.panel{background:white;border:1px solid #dce4ed;border-radius:14px;padding:22px;margin:20px 0}.note{background:#fff3d9;border-left:4px solid #bc831d;padding:14px 18px}a{color:#2454d0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:12px 8px;border-bottom:1px solid #e0e6ed;text-align:left;vertical-align:top}.scroll{overflow:auto}code{font-size:12px;overflow-wrap:anywhere}.muted{color:#59697b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%;height:auto;display:block}select{font:inherit;padding:8px;border:1px solid #becbda;border-radius:6px}.badge{background:#eef2f8;border-radius:6px;padding:5px 10px;font-size:12px}#proof{font-weight:600}.red{color:#ae3939}.green{color:#137957}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:20px 12px}h1{font-size:25px}}</style>
<main><p class="muted">RULE2REGION / BOUNDED REVISION V3 / 2026-10-11</p><h1>有限线要横切通道，来源字段不能缺席</h1><div class="note"><b>仅补修已复现 I1 / M1 · 新 head 独审 PENDING</b><br>850b4f3 独审定级为 C0/I1/M1，独立软件 38 项通过。v2 的 10 项浏览器检查仅属作者证据。当前不宣称准入完备，也没有模型识门或真实视觉实验。</div>
<section class="panel"><h2>保留旧假准入，单列新拒绝结果</h2><div class="scroll"><table><thead><tr><th>反例及最终分级</th><th>旧 v2 结果</th><th>v3 结果</th></tr></thead><tbody>__ROWS__</tbody></table></div><p><a href="../building_boundary_v2/report_v2.html">原 v2 报告</a> · <a href="../building_boundary/report.html">原 v1 报告</a> · <a href="results_v3.json">完整增量结果与反例</a> · <a href="RED_PHASE.txt">先行红灯测试日志</a></p><p class="muted">来源缺失反例计数误差为 0，仍必须拒绝：数值碰巧正确不能替代可追溯来源。旧有限夹具成绩不改写。</p></section>
<section class="panel"><h2>审核走廊与有限横切</h2><label>几何例子 <select id="example"><option value="short">短线：只与走廊相交</option><option value="endpoint">不同线：实际连接两侧边界</option><option value="tilted">不同倾斜线：完整横切</option></select></label><div class="grid"><svg id="scene" viewBox="0 0 550 360" role="img" aria-label="合成矩形通道与有限计数线"></svg><div><p id="proof"></p><p>灰色上下边界是作者审核声明的不可绕行边界；左右侧分别通往区域 a / b。绿色路径是空间示意，不来自事件 GT。</p><p>计数线必须在两个相对边界上都有实际交点，有限线参数在 [0,1]，交点位于边界内部；ROI 覆盖完整走廊。把短线无限延长后分开锚点，不证明有限线分割通道。</p><p class="muted">只支持此处明确声明的轴对齐矩形 / 不可绕行边界抽象。未知形状、可绕行或缺现场证据保持 UNKNOWN；此图是自制坐标参考，不是现场核验。</p></div></div></section>
<section class="panel"><h2>来源先做 schema，再比较值</h2><p>来源 <code>source_id / revision</code> 与会话 <code>source_id / source_revision</code> 各自必填：ID 为去除空白后非空字符串，版本为严格正整数，排除 bool。通过存在、非空、类型与范围校验后才比较一致性。</p><p>双方删除、null、空白字符串、数字 ID、非正数版本、bool 或字符串版本，即便重写 hash 和模拟收据仍拒绝。配置 / 会话资格收据同时升级绑定 admission-v3。</p></section>
<section class="panel"><h2>可保留的有限证据与未知</h2><ul><li>两个新反例已复现并被 v3 拒绝；被拒运行没有正式预测和成绩。</li><li>两个不同完整几何正例保留：[[.49,.25],[.51,.75]] 与 [[.46,.2],[.54,.8]]，原固定执行器下六目标误差为 0。</li><li>准入不读取人物轨迹或事件 / 占用 GT；只在评分阶段读 GT。门、连接区域、室外边界和现场参考仍预置。</li><li>模型候选、模型 + 元数据、真实视觉均 NOT_RUN；真实分数 null。</li></ul><p><span class="badge">新 head 独审 PENDING</span> <span class="badge">当前不称准入完备</span></p><p><a href="PROTOCOL_V3.md">有界修正协议</a> · <a href="README.md">复现与适用限制</a> · <a href="MANIFEST.json">增量 manifest</a> · <a href="../building_boundary_v2/PROTOCOL_V2.md">下一阶段最小真实输入与模块拆分</a></p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);
const lines={short:[[.5,.26],[.5,.30]],endpoint:[[.49,.25],[.51,.75]],tilted:[[.46,.2],[.54,.8]]};
function draw(){const type=$('example').value,line=lines[type],x=v=>45+(v-.2)/.6*460,y=v=>35+(v-.1)/.8*280;
let svg=`<rect x="${x(.45)}" y="${y(.25)}" width="${x(.55)-x(.45)}" height="${y(.75)-y(.25)}" fill="#e7f3f0"/><line x1="${x(.45)}" y1="${y(.25)}" x2="${x(.55)}" y2="${y(.25)}" stroke="#59697b" stroke-width="7"/><line x1="${x(.45)}" y1="${y(.75)}" x2="${x(.55)}" y2="${y(.75)}" stroke="#59697b" stroke-width="7"/><text x="150" y="175" font-size="16">b 区域</text><text x="340" y="175" font-size="16">a 区域</text><path d="M ${x(.7)} ${y(.5)} L ${x(.3)} ${y(.5)}" stroke="#16876c" stroke-width="2" stroke-dasharray="6 4"/><text x="${x(.45)-10}" y="${y(.25)-15}" font-size="12">不可绕行边界</text><line x1="${x(line[0][0])}" y1="${y(line[0][1])}" x2="${x(line[1][0])}" y2="${y(line[1][1])}" stroke="#b43939" stroke-width="5"/>`;
line.forEach(p=>svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="5" fill="#b43939"/>`);$('scene').innerHTML=svg;
$('proof').textContent=type==='short'?'拒绝：短有限线未连接两侧边界；旧 v2 漏 3 次，整栋 max=1。':'作者软件验证准入：有限线构成完整横切；原有限夹具误差 0。';$('proof').className=type==='short'?'red':'green';}
$('example').addEventListener('change',draw);draw();
</script></html>'''
    (ROOT/'report_v3.html').write_text(html.replace('__ROWS__',rows).replace('__DATA__',payload))
    print('report_v3.html generated; previous versions unchanged')

if __name__=='__main__':main()
