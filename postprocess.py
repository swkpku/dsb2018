file_name = 'resnet50__lr1_bs128_size75_epoch_4_iter_10_singlecrop_top5.csv'
new_file_name = 'resnet50__lr1_bs128_size75_epoch_4_iter_10_singlecrop_top5_postprocess.csv'

fout = open(new_file_name, 'w')
fout.write('id,is_iceberg\n')

with open(file_name, 'r') as f:
    for line in f:
        Id = line.split(' ')[0]
        prob1 = float(line.split(' ')[1].split(':')[1])
        prob2 = float(line.split(' ')[2].split(':')[1])
        
        new_prob1 = prob1
        if new_prob1 < 0:
            new_prob1 = 0
        elif new_prob1 > 1:
            new_prob1 = 1
        #new_prob2 = (prob2 - min(prob1, prob2)) / (max(prob1, prob2) - min(prob1,prob2))

        fout.write(Id+','+str(new_prob1)+'\n')
    
        
