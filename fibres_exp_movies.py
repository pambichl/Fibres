import subprocess
import numpy as np

file_mat = '0925_101lambda_256k'
file_dir = '../Data-Numerics/' + file_mat + '/'

lambdas = np.load(file_dir + file_mat + '_wavelength.npy')
N_l = np.shape(lambdas)[0]

### params ###
s_rat = 0.030
cost = 1
CMmax = 1e7
CMmax = None
mov = 1
### ###

pics_name = file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat) + file_mat + '.s=%.3f.*.jpg' % s_rat
movie_name =  file_dir + "cost=%i/movie/s=%.3f/" % (cost, s_rat) + "speckles." + file_mat

subprocess.call(["mencoder",
                 "mf://" + pics_name,
                 "-mf", "w=1500:h=600:fps=%f:type=jpg" % (N_l/15.0),
                 "-ovc", "lavc",
                 "-of", "avi",
                 "-o", movie_name + ".avi"])

subprocess.call(["ffmpeg", "-i", movie_name + ".avi", "-s", "1280x960", "-b", "1000k", "-vcodec", "wmv2", "-ar" ,"44100", "-ab", "56000", "-ac", "2", "-y", movie_name + ".wmv"])


