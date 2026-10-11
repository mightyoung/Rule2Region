"""Generate a self-contained interactive synthetic report from frozen inputs."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def main():
    names = ['results','topology','goals','states_gt','events_gt','observations','admission']
    data = {n:json.loads((ROOT/(n+'.json')).read_text()) for n in names}
    template = (ROOT/'report_template.html').read_text()
    payload = json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (ROOT/'report.html').write_text(template.replace('__DATA__',payload))
    print('report.html generated: embedded synthetic inputs; no external requests')

if __name__ == '__main__':
    main()
