#!/bin/bash

# stimuli
wget https://www.dropbox.com/s/55ei85jhlz5b3g1/stimuli_180concepts.txt?dl=1
wget https://www.dropbox.com/s/jtqnvzg3jz6dctq/stimuli_384sentences.txt?dl=1
wget https://www.dropbox.com/s/qdft8l21e83kgz0/stimuli_243sentences.txt?dl=1

# data
wget https://www.dropbox.com/s/5umg2ktdxvautci/P01.tar?dl=1
wget https://www.dropbox.com/s/04rcipsc1zi35j6/M01.tar?dl=1
wget https://www.dropbox.com/s/parmzwl327j0xo4/M02.tar?dl=1
wget https://www.dropbox.com/s/56uskp7gl4unehy/M03.tar?dl=1
wget https://www.dropbox.com/s/4p9sbd0k9sq4t5o/M04.tar?dl=1
wget https://www.dropbox.com/s/a1s0qgj6mdfrdqy/M05.tar?dl=1
wget https://www.dropbox.com/s/h0je2e3eiud035w/M06.tar?dl=1
wget https://www.dropbox.com/s/4gcrrxmg86t5fe2/M07.tar?dl=1
wget https://www.dropbox.com/s/3q6xhtmj611ibmo/M08.tar?dl=1
wget https://www.dropbox.com/s/kv1wm2ovvejt9pg/M09.tar?dl=1
wget https://www.dropbox.com/s/tffxs9sddgw9wep/M10.tar?dl=1
wget https://www.dropbox.com/s/pps2nude3dhjegw/M13.tar?dl=1
wget https://www.dropbox.com/s/8i0r88n3oafvsv5/M14.tar?dl=1
wget https://www.dropbox.com/s/swc5tvh1ccx81qo/M15.tar?dl=1
wget https://www.dropbox.com/s/tjs4gasm8fjgek1/M16.tar?dl=1
wget https://www.dropbox.com/s/vruzr4fiytqcen4/M17.tar?dl=1

## untar
for f in *.tar*
    do tar -xvf ${f}
    echo "rm ${f}"
done
