"""Author UI evidence only; requires already installed Playwright/Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':900})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content((ROOT/'report_v3.html').read_text())
    text=page.locator('main').inner_text()
    assert 'C0/I1/M1' in text and '新 head 独审 PENDING' in text
    checks.append('review severity and pending new independent review displayed')
    assert 'Important' in text and 'Minor' in text and 'max 1 / FN 3' in text
    checks.append('preserved v2 false admissions and current no-score rejections displayed')
    assert '拒绝' in page.locator('#proof').inner_text()
    short=page.locator('#scene').inner_html()
    page.select_option('#example','endpoint')
    assert '完整横切' in page.locator('#proof').inner_text() and page.locator('#scene').inner_html()!=short
    page.select_option('#example','tilted')
    assert '完整横切' in page.locator('#proof').inner_text()
    checks.append('short and two different full finite cuts interactive')
    assert 'NOT_RUN' in text and '准入完备' in text
    checks.append('uncalled models and bounded-completeness limitation visible')
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path='/tmp/building_boundary_v3_desktop.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    checks.append('390px layout without document horizontal overflow')
    assert not errors,errors
    browser.close()
evidence={'status':'PASS','evidence_role':'author_UI_checks_not_independent_review',
    'checks':checks,'page_errors':errors,'render_method':'local self-authored HTML via set_content'}
(ROOT/'browser_checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
