import sys

for line in sys.stdin:
    if line.strip() == "":
        continue
    else:
        line = line.replace(u'\ufeff', '')
        line = " ".join(line.split())
        sys.stdout.write(line + "\n")
