batch_size = 25 ## adjust this number to fit in memory capacity of your GPU
num_action = 100 ## number of candidates push actions to be sampled from current image, the number should be a multiple of batch_size

model_path ="/home/tomo/ws_tomo/src/command/tomo_imaging/src/vision/pushnet/model"

image_resolution = {"width": 128.0, "height": 106.0}

mode ={"position": "xy", 
       "rotation": "w",
       "reconfigure" : "wxy"}

method = {"with_COM":"simcom", 
          "without_COM" : "sim", 
          "without_memory" : "nomem"}

target_pose = "/home/tomo/ws_tomo/src/command/tomo_imaging/src/vision/pushnet/data/target_pose.png"#"./data/target_pose.png"

### three differetn network architecture for comparison
arch = {
        'simcom':'push_net',
        'sim': 'push_net_sim',
        'nomem': 'push_net_nomem'
       }



