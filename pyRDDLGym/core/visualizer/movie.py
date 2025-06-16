import glob
import numpy as np
import os
from PIL import Image
import re
from typing import Any, Iterable, Optional

from pyRDDLGym.core.debug.exception import raise_warning

# (mike: #166) opencv is now optional
try:
    import cv2
    _ALLOW_MP4 = True
except:
    raise_warning('cv2 is not installed: save_as_mp4 option will be disabled.', 'red')
    _ALLOW_MP4 = False


class ImageWriter:
    '''Class for writing images to disk.'''

    def __init__(self,
                 save_dir: str,
                 env_name: str,
                 max_frames: int,
                 skip: int=1,
                 save_format: str='png') -> None:
        '''Creates a new image writer for writing images to disk.
        
        :param save_dir: the directory to save images to
        :param env_name: the root name of each image file
        :param max_frames: the max number of frames to save
        :param skip: how often frames should be recorded
        :param save_format: the format in which to save individual frames
        '''
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, env_name + '_{}_temp' + '.' + save_format)
        self.env_name = env_name
        self.max_frames = max_frames
        self.skip = skip
        
        self._n_frame = 0
        self._time = 0

    def reset(self) -> None:
        '''Removes all image files currently saved to disk.'''
        load_path = self.save_path.format('*')
        files = glob.glob(load_path)
        removed = 0
        for file in files:
            os.remove(file)
            removed += 1
        if removed:
            raise_warning(f'Removed {removed} temporary files at {load_path}.')
        self._n_frame = 0
        self._time = 0

    def save_frame(self, image: Any) -> None:
        '''Saves the given image as a file on disk.'''
        if self._n_frame >= self.max_frames:
            return     
        if self._time % self.skip != 0: 
            self._time += 1
            return
        
        file_path = self.save_path.format(str(self._n_frame).rjust(10, '0'))
        image.save(file_path)
        self._n_frame += 1 
        self._time += 1

    def load_frames(self) -> Iterable[Any]:
        '''Loads the images saved on disk as a sequences of PIL images.'''
        load_path = self.save_path.format('*')

        def getOrder(frame):
            return int(re.search('\d+', frame).group(0))

        files = glob.glob(load_path)
        files.sort(key=getOrder)
        images = map(Image.open, files)
        return images


class MovieGenerator:
    
    def __init__(self,
                 save_dir: str,
                 env_name: str,
                 max_frames: int,
                 skip: int=1,
                 save_format: str='png',
                 frame_duration: int=100,
                 loop: int=0,
                 save_as_mp4: bool=False) -> None:
        '''Creates a new movie generator for creating movies out of still frames.

        :param save_dir: the directory to save images to
        :param env_name: the root name of each image file
        :param max_frames: the max number of frames to save
        :param skip: how often frames should be recorded
        :param save_format: the format in which to save individual frames
        :param frame_duration: the duration of each frame in the animated video
        :param loop: how many times the animated GIF should loop
        :param save_as_mp4: whether to save mp4 video (or GIF if False)
        '''
        self.writer = ImageWriter(save_dir, env_name, max_frames, skip, save_format)
        self.env_name = env_name
        self.frame_duration = frame_duration
        self.loop = loop
        self.save_as_mp4 = save_as_mp4
    
    def save_frame(self, image: Any) -> None:
        self.writer.save_frame(image)

    def save_animation(self, file_name: Optional[str]=None) -> None:
        if _ALLOW_MP4 and self.save_as_mp4:
            self.save_mp4(file_name)
        else:
            self.save_gif(file_name)
        self.writer.reset()
            
    def save_gif(self, file_name: Optional[str]=None) -> None:
        if file_name is None:
            file_name = self.writer.env_name
        images = self.writer.load_frames()  

        save_path = os.path.join(self.writer.save_dir, file_name + '.gif')
        frame0 = next(images, None)
        if frame0 is not None:
            frame0.save(fp=save_path,
                        format='GIF',
                        append_images=images,
                        save_all=True,
                        duration=self.frame_duration,
                        loop=self.loop)  
        
    def save_mp4(self, file_name: Optional[str]=None) -> None:
        if file_name is None:
            file_name = self.writer.env_name
        images = self.writer.load_frames()

        video_writer, w, h = None, None, None
        fps = 1000 / self.frame_duration
        for file in images:
            frame = cv2.imread(file)
            if w is None:
                h, w, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                video_writer = cv2.VideoWriter(file_name + '.mp4', fourcc, fps, (w, h))
            video_writer.write(frame)
        if video_writer is not None:
            video_writer.release()


class CompositeFrameGenerator:

    def __init__(self, save_dir: str,
                 env_name: str,
                 max_frames: int,
                 skip: int=1,
                 save_format: str='png', 
                 output_format: str='png'):
        '''Creates a writer for creating composite (averaged) frames out of still frames.

        :param save_dir: the directory to save images to
        :param env_name: the root name of each image file
        :param max_frames: the max number of frames to save
        :param skip: how often frames should be recorded
        :param save_format: the format in which to save individual frames
        :param output_format: the format in which to save the composite
        '''
        self.writer = ImageWriter(save_dir, env_name, max_frames, skip, save_format)
        self.env_name = env_name
        self.output_format = output_format
    
    def save_frame(self, image: Any) -> None:
        self.writer.save_frame(image)
        
    def save_animation(self, file_name: Optional[str]=None) -> None:
        self.save_composite(file_name)
        self.writer.reset()
            
    def save_composite(self, file_name: Optional[str]=None) -> None:
        if file_name is None:
            file_name = self.writer.env_name
        images = self.writer.load_frames()  

        # average the image pixels arithmetically
        arr = 0.0
        count = 0
        for image in images:
            img_arr = np.array(image, dtype=float)
            count += 1
            arr = arr + (img_arr - arr) / count    
        avg_img = Image.fromarray(np.array(np.round(arr), dtype=np.uint8))

        save_path = os.path.join(self.writer.save_dir, file_name + '.' + self.output_format)
        avg_img.save(save_path)
