# run this script from the octave prompt
#load data.log
load data_painting_annoying.log
data = data_painting_annoying ;

N = size(data)(2);
t = data(:,1);
data = data(:,2:N);
N = N - 1;

data(:,3) = -50+100*data(:,3);
data(:,6) = 20+40*data(:,6);
data(:,8) = 100*data(:,8);

labels = [
          "-;x;"
          "-;y;"
          "-;pressure;"
          "-;v;"
          "-;v filtered;"
          "-*;signal;"
          "-;slow_v;"
          "-;slow_v_ratio;"
          ];

grid on
plot(t, data, labels)
