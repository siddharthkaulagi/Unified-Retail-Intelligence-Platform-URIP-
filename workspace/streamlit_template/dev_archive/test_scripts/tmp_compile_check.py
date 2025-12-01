import os
files = [
    r"c:\Users\sidda\Downloads\Retail Sales Prediction\workspace\streamlit_template\pages\4_📋_Reports.py",
    r"c:\Users\sidda\Downloads\Retail Sales Prediction\workspace\streamlit_template\utils\models.py",
    r"c:\Users\sidda\Downloads\Retail Sales Prediction\workspace\streamlit_template\pages\3_📈_Dashboard.py",
    r"c:\Users\sidda\Downloads\Retail Sales Prediction\workspace\streamlit_template\pages\2_🔮_Model_Selection.py",
]

for path in files:
    print('Checking', path)
    if not os.path.exists(path):
        print('MISSING', path)
        continue
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    try:
        compile(s, path, 'exec')
        print('COMPILE_OK')
    except Exception as e:
        print('COMPILE_ERROR', path)
        import traceback; traceback.print_exc()
