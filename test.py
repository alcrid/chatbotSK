
import csv

file = []
with open("Final_version/model_history_log.csv") as f:
   data = f.read()

final = []
for i,line in enumerate(data.split("\n")):
    if(i != 0):
        tmp = line.split(",")
        tmp[0] = i
        final.append(tmp)
    else:
        tmp = line.split(",")
        final.append(tmp)

with open("Final_version/model_history_log_final.csv","w",newline='') as f:
   writer = csv.writer(f)
   for line in final:
       writer.writerow(line)



print(final)