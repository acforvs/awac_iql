## Offline RL

В этом репозитории представлены два алгоритма:
* AWAC (awac.py, awac.ipynb)
* IQL (iql.py, iql.ipynb)


.ipynb файлы рекомендуется запускать в Google colab

Для успешной работы .py на локальной машине должен быть настроен d4rl и mujoco.

Для этого можно попробовать сделать следующее:
```
 apt-get -qq update
 apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
 mkdir ~/.mujoco
 wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
 tar -zxf mujoco.tar.gz -C "$HOME/.mujoco"
 rm mujoco.tar.gz
 echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
 echo 'export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
 echo "/root/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf
 ldconfig
 pip3 install -U 'mujoco-py<2.2,>=2.1'
 touch .mujoco_setup_complete
```

Если после этого будет не хватать каких-то прописанных значений в bashrc, код при запуске должен упасть с ошибкой, которая говорит, как это можно исправить

Скрипты для настройки локальной машины не предоставляются, можно попробовать воспользоваться https://github.com/tinkoff-ai/CORL/blob/main/Dockerfile


## Запуск обучения

Чтобы запустить обучение локально, можно использовать команду
```
python3 iql.py --yaml_path=configs/iql/antmaze_medium_seed_0.yaml
```
В ``yaml_path`` через аргумент командной строки может передаваться путь до .yaml с конфигурацией


Реализация вдохновлена https://github.com/tinkoff-ai/CORL и имплементациями:
* https://github.com/ikostrikov/jaxrl
* https://github.com/ikostrikov/implicit_q_learning
* https://github.com/rail-berkeley/rlkit/tree/master/examples/iql
* https://github.com/Manchery/iql-pytorch/tree/master
