import glob
import os
from PIL import Image
import warnings
import re

# (mike: #166) opencv is now optional
try:
    import cv2
    _ALLOW_MP4 = True
except:
    warnings.warn('cv2 is not installed: save_as_mp4 option will be disabled.',
                  stacklevel=2)
    _ALLOW_MP4 = False


class MovieGenerator:
    
    def __init__(self,
                 save_dir: str,
                 env_name: str,
                 max_frames: int,
                 skip: int=1,
                 save_format: str='png',
                 frame_duration: int=100,
                 loop: int=0,
                 save_as_mp4: bool=False):
        '''Creates a new movie generator for saving frames to disk, and creating animated GIFs.
        
        :param save_dir: the directory to save images to
        :param env_name: the root name of each image file
        :param max_frames: the max number of frames to save
        :param skip: how often frames should be recorded
        :param save_format: the format in which to save individual frames
        :param frame_duration: the duration of each frame in the animated video
        :param loop: how many times the animated GIF should loop
        :param save_as_mp4: whether to save mp4 video (or GIF if False)
        '''
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, env_name + '_{}' + '.' + save_format)
        self.env_name = env_name
        self.max_frames = max_frames
        self.skip = skip
        self.frame_duration = frame_duration
        self.loop = loop
        self.save_as_mp4 = save_as_mp4
        
        self._n_frame = 0
        self._time = 0
    
    def reset(self) -> None:
        load_path = self.save_path.format('*')
        files = glob.glob(load_path)
        removed = 0
        for file in files:
            os.remove(file)
            removed += 1
        if removed:
            warnings.warn(f'removed {removed} temporary files at {load_path}',
                          stacklevel=2)
        
        self._n_frame = 0
        self._time = 0
        
    def save_frame(self, image) -> None:
        if self._n_frame >= self.max_frames:
            return     
        if self._time % self.skip != 0: 
            self._time += 1
            return
        
        file_path = self.save_path.format(
            str(self._n_frame).rjust(10, '0'))
        image.save(file_path)
        self._n_frame += 1 
        self._time += 1
    
    def save_animation(self, file_name: str=None):
        if _ALLOW_MP4 and self.save_as_mp4:
            self.save_mp4(file_name)
        else:
            self.save_gif(file_name)
        self.reset()
            
    def save_gif(self, file_name: str=None):
        if file_name is None:
            file_name = self.env_name
        load_path = self.save_path.format('*')

        def getOrder(frame):
            return int(re.search('\d+', frame).group(0))

        files = glob.glob(load_path)
        files.sort(key=getOrder)

        # images = map(Image.open, glob.glob(load_path))
        images = map(Image.open, files)
        
        save_path = os.path.join(self.save_dir, file_name + '.gif')
        frame0 = next(images, None)
        if frame0 is not None:
            frame0.save(fp=save_path,
                        format='GIF',
                        append_images=images,
                        save_all=True,
                        duration=self.frame_duration,
                        loop=self.loop)  
        
    def save_mp4(self, file_name: str=None):
        if file_name is None:
            file_name = self.env_name
        load_path = self.save_path.format('*')
        images = glob.glob(load_path)
        
        writer, w, h = None, None, None
        fps = 1000 / self.frame_duration
        for file in images:
            frame = cv2.imread(file)
            if w is None:
                h, w, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                writer = cv2.VideoWriter(file_name + '.mp4', fourcc, fps, (w, h))
            writer.write(frame)
        if writer is not None:
            writer.release()
