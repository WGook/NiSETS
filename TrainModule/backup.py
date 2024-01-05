import os
import shutil

def file_backup(
            prtpath = '.',
            run_file_dir = 'run_code',
            ckpt_dir_name = 'ckpt'
            ):
    filelist = os.listdir(prtpath)
    for dir in ['wandb', 'checkpoint']:
        if dir in filelist:
            filelist.remove(dir)
    code_path = run_file_dir
    os.mkdir(code_path)
    print('>>> backup dir is made in \n{}'.format(run_file_dir))

    for file in filelist:
        path = os.path.join(prtpath, file)
        if os.path.isdir(file):
            shutil.copytree(path, os.path.join(code_path, file))
        elif os.path.isfile(file):
            shutil.copy(path, os.path.join(code_path, file))
    print('>>> run files are saved in \n{}'.format(run_file_dir))

    ckpt_path = os.path.join(code_path, ckpt_dir_name)
    os.mkdir(ckpt_path)
    print('>>> ckpt dir is made in \n{}'.format(ckpt_path))
    return run_file_dir, ckpt_path
