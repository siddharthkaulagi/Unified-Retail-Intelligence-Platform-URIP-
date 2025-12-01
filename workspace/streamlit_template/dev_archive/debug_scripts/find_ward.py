with open('bbmp_final_new_wards.kml', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Atturu' in line:
            print(f"Found at line {i}:")
            for j in range(max(0, i-5), min(len(lines), i+5)):
                print(lines[j].strip())
            break
