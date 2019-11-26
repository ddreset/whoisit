import numpy as np
import os
import momentnet
import random
import tensorflow as tf
import json
import dataformat
import shutil
import crop_mosaic
from array import array
from datetime import datetime

weight_set_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weight_sets")
# content of all set.json
weight_sets = []


def discover_weight_set():
    weight_sets.clear()
    for name in os.listdir(weight_set_path):
        pwd = os.path.join(weight_set_path, name)
        set_file = os.path.join(pwd, "set.json")
        if os.path.isdir(pwd) and os.path.exists(set_file):
            with open(set_file, "r") as file:
                set_data = json.load(file)
                weight_sets.append(set_data)


discover_weight_set()


template_set_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")
# name of all folders in ./templates
template_paths = []


def discover_template_set():
    template_paths.clear()
    for name in os.listdir(template_set_path):
        pwd = os.path.join(template_set_path, name)
        if os.path.isdir(pwd):
            template_paths.append(name)


discover_template_set()


class Runner:

    def __init__(self):
        self.running = False
        # start with weight_sets[0], get a network with the chosen weights
        self.setup(0)
        # start with template[0], get pic data array and label array
        self.change_template(0)
        self.collect_template_flag = -1

    def change_template(self, template_index):
        discover_template_set()
        if len(template_paths) < template_index:
            return False
        self.template_index = template_index
        self.templates, self.template_labels = dataformat.read_template_directory(self.formatter, os.path.join(template_set_path, template_paths[self.template_index]), with_flip=True)
        print(template_paths[self.template_index])
        return True

    def use_temporary_template(self, template_list):
        if template_list is None:
            return False
        
        self.templates, self.template_labels, self.label_dict = dataformat.read_template_urls(self.formatter, template_list)
            
    def setup(self, index):
        self.index = index
        if len(weight_sets) <= self.index:
            return False
        s = weight_sets[self.index]["size"] # e.g. [256,32]
        self.size = (s[0], s[1])
        self.num_layers = weight_sets[self.index]["num_layers"]
        self.session_name = "weight_sets/" + weight_sets[self.index]["session_name"]

        # get contour data and labels of pics from /data folder
        self.formatter = dataformat.DataFormat(self.size[0])

        # close the running session, because we are gonna start a new one.
        self.close_down()

        print(self.session_name)
        # initialize model
        num_intra_class = 10
        num_inter_class = 20
        self.comparator = momentnet.Comparator((2, self.size[0]), self.size[1], num_intra_class=num_intra_class, num_inter_class=num_inter_class, layers=self.num_layers)

        # start new session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.running = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # load the chosen session = load the chosen weights
        self.comparator.load_session(self.sess, self.session_name)
        return True

    def process(self, frames):
        if not self.running:
            return None

        if isinstance(frames, list): # a few pictures
            data = np.zeros([len(frames), 2, self.size[0]], dtype=np.float32)
            for i, frame in enumerate(frames):
                data[i, ...] = self.formatter.format(frame)
            c, raw = self.comparator.process(self.sess, np.reshape(data, [-1, self.size[0] * 2]), np.reshape(self.templates, [-1, self.size[0] * 2]))
            raw = raw[:, c].flatten()
            classes = self.template_labels[c, 0]
            flip_or_not = self.template_labels[c, 1]

        else: # one picture
            frame = frames
            # extract contour data pic
            data = self.formatter.format(frame)
            # predict class and raw confidence
            c, raw = self.comparator.process(self.sess, np.reshape(data, [-1, self.size[0] * 2]), np.reshape(self.templates, [-1, self.size[0] * 2]))
            print("raw:")
            print(raw)
            raw = raw[:, c].flatten()
            classes = self.template_labels[c, 0]
            flip_or_not = self.template_labels[c, 1]
            # if template flag is set, upload pic to chosen template folder with proper filename.
            if self.collect_template_flag >= 0:
                print(self.collect_template_flag)
                dataformat.write_to_template_directory(frame, random.randint(0, 100000), self.collect_template_flag, self.formatter, template_paths[self.template_index])
                self.collect_template_flag = -1

        return classes, raw, flip_or_not

    def raise_template_flag(self, label):
        print(label)
        self.collect_template_flag = int(label)

    def close_down(self):
        if self.running:
            self.sess.close()
            tf.reset_default_graph()
        self.running = False

    def get_weight_sets(self):
        discover_weight_set()
        return weight_sets, self.index

    def get_template_sets(self):
        discover_template_set()
        return template_paths, self.template_index

    def archive_selected(self):
        artifact_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "artifacts")
        pack_dir = os.path.join(artifact_dir, "pack")
        if os.path.exists(pack_dir):
            shutil.rmtree(os.path.join(artifact_dir, "pack"))
        os.makedirs(os.path.join(artifact_dir, "pack"))
        weight_path = "/".join(weight_sets[self.index]["session_name"].split("/")[:-1])
        print(weight_path)
        shutil.copytree(os.path.join(weight_set_path, weight_path), os.path.join(artifact_dir, "pack", weight_path))
        shutil.copytree(os.path.join(template_set_path, template_paths[self.template_index]), os.path.join(artifact_dir, "pack", "templates"))
        shutil.make_archive(os.path.join(artifact_dir, "pack"), 'gztar', os.path.join(artifact_dir, "pack"))
        return os.path.join(artifact_dir, "pack.tar.gz")

    def save_weight_for_web_download(self):
        if not self.running:
            return None

        weights = self.comparator.get_weights(self.sess, np.reshape(self.templates, [-1, self.size[0] * 2]))
        weights.append(self.template_labels[:, 0])

        web_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")
        model_dir = os.path.join(web_dir, "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        write_model_weight(web_dir, weights, "model/model")

    def count_templates(self, folder_name):
        folder_path = os.path.join(template_set_path, folder_name)
        if (not os.path.exists(folder_path)) or (not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
            # update template sets
            discover_template_set()
        file_num = len({name.split(".")[0] for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))})
        return file_num

    def update_templates_folder(self, template_name, templates_list, cropper):
        folder_path = os.path.join(template_set_path, template_name)
        if (not os.path.exists(folder_path)) or (not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
            discover_template_set()
        # write in to log file
        with open(template_set_path+"/"+template_name+".txt", "w") as file:
            json.dump({
                "templates":templates_list,
                "date_created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, file, sort_keys=True, indent=4)

        label = 1
        for template in templates_list:
            # crop
            img = crop_mosaic.read_image(template["thumbnail"])
            if img is not None:
                cropped_img,__ = crop_mosaic.crop(cropper.model,img)
                # save images/image and update log 
                counter = 0
                for cropped in cropped_img:
                    cropped.save(folder_path+"/"+str(counter) +"."+str(label)+".0,0,100,100.png")
                    counter = counter + 1
            label = label + 1

def write_model_weight(root, weights, name):
    outfile_name = name + ".json"
    with open(os.path.join(root, outfile_name), 'w') as outfile:

        data = []
        for i, w in enumerate(weights):
            if isinstance(w, list):
                child_file_name = write_model_weight(root, w, name + "_" + str(i))
                data.append({"t": "n",
                             "path": child_file_name})
            else:
                child_file_name = name + "_" + str(i) + ".bin"
                with open(os.path.join(root, child_file_name), 'wb') as child_output:
                    if w.dtype == np.int32:
                        array('i', w.astype(np.int32).flatten().tolist()).tofile(child_output)
                        data.append({"t": "i",
                                     "shape": w.shape,
                                     "path": child_file_name})
                    else:
                        array('f', w.astype(np.float32).flatten().tolist()).tofile(child_output)
                        data.append({"t": "f",
                                     "shape": w.shape,
                                     "path": child_file_name})

        print("writing to ", outfile_name)
        json.dump(data, outfile)
    return outfile_name


if __name__ == '__main__':

    for weight in weight_sets:
        print(weight)

    for path in template_paths:
        print(path)
