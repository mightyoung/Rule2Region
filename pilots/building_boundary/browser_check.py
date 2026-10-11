import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root = Path(__file__).resolve().parent
errors = []
checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.set_content((root/'report.html').read_text(), wait_until='load')
    page.wait_for_selector('#cards .card')
    assert page.locator('#cards .card').count() == 6
    assert 'SYNTHETIC_ADMITTED' in page.locator('#status').inner_text()
    checks.append('default six targets admitted and independent GT shown')
    page.locator('#step').fill('10')
    page.locator('#step').dispatch_event('input')
    assert 'p3' in page.locator('#eventText').inner_text() and 'p4' in page.locator('#eventText').inner_text()
    checks.append('step 10 simultaneous GT transitions visible')
    page.select_option('#goal','floor1')
    page.select_option('#door','east')
    assert page.locator('#cameras .cam').count()==2
    assert '责任镜头' in page.locator('#cameras').inner_text() and '不累加' in page.locator('#cameras').inner_text()
    checks.append('same door two views with unique owner visible')
    page.screenshot(path='/tmp/building_boundary_report_desktop.png',full_page=True)
    page.select_option('#strategy','model_candidate')
    assert 'NOT_RUN' in page.locator('#status').inner_text()
    assert 'NOT_SCORED' in page.locator('#metricTable').inner_text()
    assert '—' in page.locator('#cards').inner_text()
    checks.append('uncalled model has no predictions or scores')
    page.select_option('#strategy','door_only')
    assert 'REJECTED' in page.locator('#status').inner_text()
    assert 'NOT_SCORED' in page.locator('#metricTable').inner_text()
    page.check('#diagnostic')
    assert '强制诊断' in page.locator('#status').inner_text()
    assert '非正式' in page.locator('#metricNote').inner_text() or '不能作为正式' in page.locator('#metricNote').inner_text()
    assert 'NOT_SCORED' not in page.locator('#metricTable').inner_text()
    checks.append('rejected door-only scores hidden until explicit diagnostic toggle')
    page.locator('#timeline button[data-step="3"]').click()
    assert page.locator('#step').input_value()=='3'
    page.locator('#topology [data-door="lift"]').click()
    assert page.locator('#door').input_value()=='lift'
    checks.append('timeline and topology click update controls')
    assert page.locator('#failures tr').count()==9
    assert page.locator('#admission tr').count()==3
    checks.append('nine failure gates and three official sources visible')
    page.set_viewport_size({'width':390,'height':844})
    page.select_option('#strategy','metadata_rule')
    page.uncheck('#diagnostic')
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path='/tmp/building_boundary_report_mobile.png',full_page=True)
    checks.append('390px mobile layout has no document horizontal overflow')
    assert not errors, errors
    browser.close()
result={'status':'PASS','browser':'installed Chromium via installed Playwright',
        'render_method':'local HTML via set_content; file URL policy blocks direct navigation', 'checks':checks,'page_errors':errors,'screenshots':'local /tmp only; self-authored synthetic report'}
(root/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
