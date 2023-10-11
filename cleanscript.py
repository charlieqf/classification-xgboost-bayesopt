import sys
import os
import re

def replace_magic_time(lines):
    new_lines = []
    for line in lines:
        match = re.match(r"get_ipython\(\).run_line_magic\('time', '(.*)'\)", line)
        if match:
            code_inside_time = match.group(1)
            new_lines.append("import time\n")
            new_lines.append("start_time = time.time()\n")
            new_lines.append(code_inside_time + "\n")
            new_lines.append("end_time = time.time()\n")
            new_lines.append('print(f"Execution time: {end_time - start_time:.2f} seconds")\n')
        else:
            new_lines.append(line)

    return new_lines


if len(sys.argv) != 2:
    print("Usage: python clean_script.py <path_to_converted_file>")
    sys.exit(1)

input_filename = sys.argv[1]
output_filename = os.path.join(os.path.dirname(input_filename), 'cleaned_' + os.path.basename(input_filename))

with open(input_filename, 'r') as f:
    lines = f.readlines()

# Removing lines that start with '# In['
cleaned_lines = [line for line in lines if not line.strip().startswith('# In[')]

cleaned_lines = replace_magic_time(cleaned_lines)

with open(output_filename, 'w') as f:
    f.writelines(cleaned_lines)

print(f"Cleaned script saved to {output_filename}")

