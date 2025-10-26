import os
#a = 'python3 lanzador_parse.py 0.1 0.0001'
alpha = [0.5*i for i in range(3,5)]
lr = [0.0002*i for i in range(1,50)]


for i in alpha:
    for j in lr:
        a = 'python3 lanzador_parse.py ' + str(i) + ' ' + str(j) + "\n"
        b = 'python3 comparador_rapido.py ' + str(i) + ' ' + str(j) + "\n"
        dir_base = os.getcwd()
        text_file = open(dir_base + "/automatico_2.sh", "a")
        text_file.write(a)
        text_file.write(b)

        # close file
        text_file.close()
