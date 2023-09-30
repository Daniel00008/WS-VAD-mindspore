import os

def edit_txt(f_write_path,txt_file,with_pre=False):
    f_write = open(f_write_path, 'wb')
    infos = []
    file=open(txt_file,'r')
    filenamelist=file.readlines()
    for line in filenamelist:
        line=line.strip('\n').strip('\r').split('/',3)[-1]
        a = '/userhome/dataset/'+line
        infos.append(a)
    info = '\n'.join(infos)
    info = info.encode()
    f_write.write(info)
    f_write.close()

if __name__ == "__main__":


    txt_file = 'list/UCF_Test.list'
    f_write_path = './UCF_Test.list'
    # ext_list = ['jpg','JPG','png','PNG','jpeg','JPEG']    
    ext_list = ['xml']        
    edit_txt(f_write_path,txt_file)    