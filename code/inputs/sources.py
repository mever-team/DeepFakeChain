import random
import pandas as pd

from json import load as load_json
from pathlib import Path
from utils import read_csv, write_csv
from dirs import DATASETS_DIR

INDEX_DIR = Path(__file__).parent / "index"
INDEX_DIR.mkdir(exist_ok=True)

class FaceForensics:
   
    categories = ("real", "deepfakes", "face2face", "faceswap", "faceshifter", "neuraltextures")
    
    cat2label = {
        "real": 0,
        "deepfakes"  : 1,
        "face2face"  : 2,
        "faceswap"   : 3,
        "faceshifter": 4,        
        "neuraltextures": 5,
    }
    
    cat2fake = {
        "real": 0,
        "deepfakes"  : 1,
        "face2face"  : 1,
        "faceswap"   : 1,
        "faceshifter": 1,        
        "neuraltextures": 1,
    }
    
    cat2dir = {
        "real": "original_sequences",
        "deepfakes"  : "manipulated_sequences/Deepfakes",
        "face2face"  : "manipulated_sequences/Face2Face",
        "faceswap"   : "manipulated_sequences/FaceSwap",
        "faceshifter": "manipulated_sequences/FaceShifter",        
        "neuraltextures": "manipulated_sequences/NeuralTextures",
    }
    
    qualities = ("raw", "hiqh", "low")
    
    quality2dir = {
        "raw" : "c0/videos",
        "high": "c23/videos",
        "low" : "c40/videos",
    }  


    def __init__(self, small_margin=True, categories=None, qualities=None, _df=None):
                
        self.index_path = INDEX_DIR / "FaceForensics++"
        self.index_path.mkdir(exist_ok=True)
            
        if _df is not None:
            self.df = _df
        
        else:
            df_path = self.index_path / ("data_1_3.pkl" if small_margin else "data_2.pkl")
            if not df_path.exists():
                print("indexing ff++ dataset for the first time")
                
                data_path = DATASETS_DIR / ("FaceForensics++_1_3" if small_margin else "FaceForensics++_2")
                    
                all_data = []
                for cat, cat_path in FaceForensics.cat2dir.items():
                    for quality, quality_path in FaceForensics.quality2dir.items():
                        video_paths = data_path / cat_path / quality_path
                        if cat == "real":
                            data = [(path, cat, quality, path.name, None) for path in video_paths.iterdir()]
                        else:
                            data = [(path, cat, quality, path.name.split("_")[0],path.name.split("_")[1]) \
                                    for path in video_paths.iterdir()]
                        all_data.append(data)            
                df = pd.concat([pd.DataFrame(data) for data in all_data], ignore_index=True)
                df.columns = ["path", "category", "quality", "base_video", "manipulation_video"]
                df.path = df.path.transform(lambda item: [p for p in Path(item).iterdir() if p.name.endswith(".png") \
                                                      or p.name.endswith(".jpg")])
                df = df.explode("path", ignore_index=True)
                df.to_pickle(df_path)
            
            self.df = pd.read_pickle(df_path)
            self.df = self.df.dropna(subset=["path"]) # SHOULD NOT NE NEEDED BUT SOME VIDEO PATHS WERE EMPTY
            
            if categories is not None:
                self.filter_categories(categories)
            if qualities is not None:
                self.filter_qualities(qualities)
    
    def __len__(self):
        return len(self.df)
        
    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("category", categories)
   
    def filter_qualities(self, qualities):
        self._filter("quality", qualities)
    
    def _create_split(self, weights):
        base_videos = self.df.base_video.unique()
        sizes = [(w * len(base_videos)) // sum(weights) for w in weights]
        sizes[-1] += len(base_videos) - sum(sizes)
        
        random.shuffle(base_videos)
        start = 0
        split_base_videos = []
        for size in sizes:
            split_base_videos.append(base_videos[start:start+size])
            start += size
        
        return split_base_videos

    def _write_split(self, path, split_base_videos):
        with open(path, "w") as f:
            for videos in split_base_videos:
                f.write(" ".join(videos) + "\n")
          
    def _read_split(self, path):
        with open(path, "r") as f:
            videos = [line.rstrip().split(" ") for line in f]
        return videos
    
    def official_split(self, mode):
        path = self.index_path / "splits" / (mode + ".json")
        with open(path, "r") as f:
            pairs = load_json(f)

        videos = []
        for pair in pairs:
            videos.extend(pair)
        videos = set(videos)
        
        df = self.df.copy()
        df = df.loc[df.base_video.transform(lambda item: item in videos)]        
        return FaceForensics(_df=df)


    def split(self, weights):
        
        weights = [int(w) for w in weights] # normalise split
        
        path = self.index_path / f"split_{'_'.join([str(w) for w in weights])}.txt"
        if not path.exists():
            print(f"indexing the split {weights} for the first time")
            split_base_videos = self._create_split(weights)
            self._write_split(path, split_base_videos)
        split_base_videos = self._read_split(path)
            
        df = self.df.copy()
        split_dfs = [df.loc[df.base_video.transform(lambda item: item in sub_base_videos)].copy() \
                     for sub_base_videos in split_base_videos]
        
        return [FaceForensics(_df=df) for df in split_dfs]
             
    def discriminative_samples(self, output_type="detection"):
                
        samples = pd.DataFrame()
        samples["path"] = self.df.path
                    
        if output_type == "detection":
            samples["output"] = self.df.category.transform(lambda item: FaceForensics.cat2fake[item])
        elif output_type == "attribution":
            samples["output"] = self.df.category.transform(lambda item: FaceForensics.cat2label[item])
        else:
            raise Exception(f"Unknown output type {output_type}, expected 'detection' or 'attribution'")
            
        return list(samples.itertuples(index=False, name=None))
    

class CelebDF:
    
    categories = ("real", "fake")
    
    def __init__(self, small_margin=True, categories=None):
        
        dset_path = DATASETS_DIR / ("Celeb-DF-v2_1_3" if small_margin else "Celeb-DF-v2_2")
        
        index_dir = INDEX_DIR / ("CelebDF_1_3" if small_margin else "CelebDF_2")
        index_dir.mkdir(exist_ok=True)
        index_path = index_dir / "index.pkl"
          
        if not index_path.exists():

            print("indexing CelebDF for the first time")
            
            data = []
            for video_path in (dset_path / "Celeb-real").iterdir():
                video_path = next(video_path.iterdir())
                label = 0
                base_id = video_path.name.split("_")[0][2:]
                swap_id = None                
                for image_path in video_path.iterdir():
                    data.append((image_path, label, base_id, swap_id))
            for video_path in (dset_path / "Celeb-synthesis").iterdir():
                video_path = next(video_path.iterdir())
                label = 1
                base_id, swap_id, _ = video_path.name.split("_")
                base_id, swap_id = base_id[2:], swap_id[2:]
                for image_path in video_path.iterdir():
                    data.append((image_path, label, base_id, swap_id))
            
            df = pd.DataFrame(data)
            df.columns = ["path", "label", "base_id", "swap_id"]
            df.to_pickle(index_path)

        self.df = pd.read_pickle(index_path)
        categories = categories or self.categories
        if "real" in categories and "fake" not in categories:
            self.filter_real()
        elif "real" not in categories and "fake" in categories:
            self.filter_fake()

    def __len__(self):
        return len(self.df)       
        
    def samples_for_detection(self):        
        return [(path, label) for path, label, _, _ in self.df.itertuples(index=False, name=None)]
                
    def samples_for_attribution(self):
        return [(path, label) for path, label, _, _ in self.df.itertuples(index=False, name=None)]
    
    def filter_real(self):
        self.df = self.df.loc[self.df["label"]==0]

    def filter_fake(self):
        self.df = self.df.loc[self.df["label"]==1]


class OpenForensics:
    
    categories = ("real", "fake")
    
    def __init__(self, mode, small_margin=True, categories=None):
        
        assert mode in ("train", "val", "test"), f"mode must be in 'train', 'val', 'test', received '{mode}'"

        if not small_margin:
            raise Exception("OpenForensics is not currently processed for large margin")
        
        if mode == "train":
            dset_path = DATASETS_DIR / "OpenForensics/Train_faces_1_3"
        elif mode == "val":
            dset_path = DATASETS_DIR / "OpenForensics/Val_faces_1_3"
        elif mode == "test":
            dset_path = DATASETS_DIR / "OpenForensics/Test-Dev_faces_1_3"
        
        self.data = []
        categories = categories or self.categories
        if "real" in categories:
            self.data.extend([(path, 0) for path in (dset_path / "real").iterdir()])
        if "fake" in categories:
            self.data.extend([(path, 1) for path in (dset_path / "fake").iterdir()])
                   
    def __len__(self):
        return len(self.data)
        
    def samples_for_detection(self):
        return self.data
                
    def samples_for_attribution(self):
        return self.data

    
    
class DFDC_preview:

    categories = ("real", "fakeA", "fakeB")
    
    cat2label = {
        "real": 0,
        "fakeA"  : 1,
        "fakeB"  : 2,
    }
    
    cat2fake = {
        "real": 0,
        "fakeA"  : 1,
        "fakeB"  : 1,
    }
    
    cat2dir = {
        "real" : "original_videos",
        "fakeA": "method_A",
        "fakeB": "method_B",
    } 
    
    dir2cat = {
        "original_videos": "real",
        "method_A": "fakeA",
        "method_B": "fakeB"
    }

    def __init__(self, small_margin=True, modes=None, categories=None):
        
        self.index_dir = INDEX_DIR / "DFDC_preview"
        self.index_dir.mkdir(exist_ok=True)
     
  
        index_path = self.index_dir / ("index_1_3.pkl" if small_margin else "index_2.pkl")

        if not index_path.exists():
            print("indexing dataset for the first time")

            original_index_path = Path(DATASETS_DIR / "DFDC/dfdc_preview_set/dataset.json")
            with open(original_index_path) as f:
                original_index = load_json(f)

            data_path = DATASETS_DIR / ("DFDC_1_3" if small_margin else "DFDC_2") / "dfdc_preview_set"

            data = []

            for video_rel_path, video_metadata in original_index.items():
                path = data_path / (video_rel_path[:-4] if video_rel_path.endswith(".mp4") else video_rel_path)
                if not path.exists(): # if this occurs, some videos have not been processed
                    continue
                cat = DFDC_preview.dir2cat[video_rel_path.split("/")[0]]

                mode = video_metadata["set"]
                context = video_rel_path.split("/")[2].split("_")[1]
                source_id = video_metadata["target_id"]
                swapped_id = video_metadata["swapped_id"]

                data.append((path, cat, mode, source_id, context, swapped_id))

            df = pd.DataFrame(data)
            df.columns = ["path", "category", "mode", "source_id", "context", "swapped_id"]          
            df.path = df.path.transform(lambda item: [p for p in Path(item).iterdir() \
                                        if p.name.endswith(".png") or p.name.endswith(".jpg")])
            df = df.explode("path", ignore_index=True) 
            df.to_pickle(index_path)    


        self.df = pd.read_pickle(index_path)
        if modes is not None:
            self.filter_modes(modes)
        if categories is not None:
            self.filter_categories(categories)
    
    
    def __len__(self):
        return len(self.df)
        
    def _filter(self, param_name, param_values):
        df = self.df
        df = pd.concat([df.loc[df[param_name]==val] for val in param_values], ignore_index=True)
        self.df = df       
    
    def filter_categories(self, categories):
        self._filter("category", categories)

    def filter_modes(self, modes):
        self._filter("mode", modes)
        
    def discriminative_samples(self, output_type="detection"):
                
        samples = pd.DataFrame()
        samples["path"] = self.df.path
                    
        if output_type == "detection":
            samples["output"] = self.df.category.transform(lambda item: self.__class__.cat2fake[item])
        elif output_type == "attribution":
            samples["output"] = self.df.category.transform(lambda item: self.__class__.cat2label[item])
        else:
            raise Exception(f"Unknown output type {output_type}, expected 'detection' or 'attribution'")
            
        return list(samples.itertuples(index=False, name=None))
    

class DFDC:
    
    categories = ("real", "fake")
    cat2label = {"real": 0, "fake": 1}  
    mode2dir = {"train": "train", "val": "validation", "test":"test"}
        
    def __init__(self, mode, categories=None, small_margin=True):
        
        dset_name = "DFDC_1_3" if small_margin else "DFDC_2"
        dset_path = DATASETS_DIR / dset_name
        dset_path = dset_path / self.__class__.mode2dir[mode]
        
        index_path = INDEX_DIR / ("DFDC_1_3" if small_margin else "DFDC_2")
        index_path.mkdir(exist_ok=True)
        index_path = index_path / mode
        
        if not index_path.exists():
            
            print(f"indexing {dset_name} mode {mode} for the first time")

            if mode == "train":
                vid2labels = {}
                for path in (DATASETS_DIR / "DFDC/train").iterdir():
                    if path.is_dir():
                        with open(path / "metadata.json", "r") as f:
                            for key, val in load_json(f).items():
                                video_name = key.split(".")[0]
                                label = int(val["label"]!="REAL")
                                vid2labels[video_name] = label
                
                data = []
                for dset_part_path in dset_path.iterdir():
                    for video_path in dset_part_path.iterdir():
                        for image_path in video_path.iterdir():
                            data.append((image_path, vid2labels[video_path.name]))
                
                write_csv(index_path, data)
            
            elif mode == "val":
                vid2labels = {}
                with open(DATASETS_DIR / "DFDC/validation/labels.csv", "r", encoding="utf8") as f:
                    _ = f.readline()  # ignore headers
                    for line in f:
                        video_name, label = line.rstrip().split(",")
                        video_name = video_name.split(".")[0]
                        label = int(label)
                        vid2labels[video_name] = label
                
                data = []
                for video_path in dset_path.iterdir():
                    for image_path in video_path.iterdir():
                        data.append((image_path, vid2labels[video_path.name]))
                write_csv(index_path, data)      

            elif mode == "test":
                vid2labels = {}
                with open(DATASETS_DIR "DFDC/test/metadata.json", "r") as f:
                    i = 0
                    for key, val in load_json(f).items():

                        video_name = key.split(".")[0]
                        label = val["is_fake"]
                        vid2labels[video_name] = label
                        i += 1
                        if i % 50 == 0:
                            print(f"\r{i}", end="")
                    print()
                print(dset_path)
                data = []
                for i, video_path in enumerate(dset_path.iterdir()):
                    print(f"\r{i}", end="")
                    for image_path in video_path.iterdir():
                        data.append((image_path, vid2labels[video_path.name]))    
                print()
                write_csv(index_path, data)   
        
        self.data = read_csv(index_path, types=(str, int))
        if categories is not None and len(categories) == 1:
            if categories[0] == "real":
                self.data = [(p, l) for p, l in self.data if l == 0]
            elif categories[0] == "fake":
                self.data = [(p, l) for p, l in self.data if l == 1]
            else:
                raise Exception("Unknown category {categories}, expected 'real' or 'fake'")
                      
    def __len__(self):
        return len(self.data)
        
    def samples_for_detection(self):
        return self.data
                
    def samples_for_attribution(self):
        return self.data

