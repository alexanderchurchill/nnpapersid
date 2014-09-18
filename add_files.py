import os,sys,time
count = 0
for pop_size in [750,1000]:
    for lim_percentage in [10,20]:
        for num_epochs in [25,50]:
            for lr in [0.05,0.1]:
                for hiddens in [100,200,300,400]:
                    for rtr in [0]:
                        for w in [10]:
                            for trial in range(0,3):
                                os.system('qsub ae_knapsack_105_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.job'.format(pop_size,lim_percentage,num_epochs,lr,hiddens,rtr,w,trial))
                                time.sleep(0.1)
                                count += 1
                                print "added:",count

