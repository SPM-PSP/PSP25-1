import os 
path="D:\\dachuan\\dataset1\\seperate\\diffusion_new_1"  # 这里换成你的文件夹路径
for directory in os.listdir(path):
    for filename in os.listdir(os.path.join(path, directory)):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(path,directory, filename)
            with open(file_path, 'r') as file:
                lines = file.read()
                lines = lines.replace('boys','')
                lines = lines.replace('girls','')
                lines = lines.replace('boy','')
                lines = lines.replace('girl','')
                lines = lines.replace('no humans','') # 删除所有 'no humans'
                lines=lines.replace("1",'')
                lines = lines.replace(lines.split(',')[0],'') #出现的地方
                lines = lines.replace(", ,",",")
                lines = lines.replace(",,",",")
            with open(file_path,'w') as file:
                write=directory.split('_')[-1] # 这里获取你需要的目录名，假设是最后一部分
                file.write(f"{write}"+lines)