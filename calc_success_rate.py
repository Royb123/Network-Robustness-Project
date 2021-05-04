import sys

"""
cmd arguments:
    1. estimate epsilon file
    2. accurate epsilon file
    or any other two you want to compare
"""
network = sys.argv[1]

estimate_file = sys.argv[1]
accurate_file = sys.argv[2] 
f = open(estimate_file, 'r')
counter_new = 0
lines_new = 0
counter_old = 0
lines_old = 0
average_ERAN_runs_new = 0
average_ERAN_runs_old = 0
total_time_new = 0
total_time_new_success = 0
average_time_new = 0
total_time_old = 0
average_time_old = 0

for line in f:
    line_str = line.split(' ')
    if line_str[0] == "using":
        continue
    lines_new += 1
    epsilon = line_str[10]
    if epsilon=="-1":
        continue
    runs = line_str[11]
    runs = int(runs[:-1])
    run_time = float(line_str[14])
    total_time_new += run_time
    if epsilon[1:3] != "-2" and epsilon[1:3] != "-3":
        counter_new += 1
        average_ERAN_runs_new += runs  # divide by counter at the end
        total_time_new_success += run_time

f.close()
f = open(accurate_file, 'r')
for line in f:
    line_str = line.split(' ')
    if line_str[0] == "using":
        continue
    lines_old += 1
    epsilon = line_str[14]
    runs = int(line_str[19])
    run_time = float(line_str[21])
    total_time_old += run_time
    if epsilon != "-1":
        counter_old += 1
        average_ERAN_runs_old += runs  # divide by counter at the end

success_percentage = (counter_new/lines_new)*100

average_ERAN_runs_new /= counter_new
average_ERAN_runs_old /= counter_old
m_new, s_new = divmod(total_time_new_success, 60)
h_new, m_new = divmod(m_new, 60)
m_old, s_old = divmod(total_time_old, 60)
h_old, m_old = divmod(m_old, 60)
print("Success rate: {per:.1f}% ({img} images out of {tot})".format(per=success_percentage, img=counter_new, tot=lines_new))
print("Average number of runs: Prediction - {pred:.1f} , Binary Search - {binar:.1f}".format\
          (pred=average_ERAN_runs_new, binar=average_ERAN_runs_old))
print("Run time : Prediction- {:.1f}:{:.2f}:{:.2f}, Binary Search- {:.1f}:{:.2f}:{:.2f}".format(h_new,m_new,s_new,h_old,m_old,s_old))

