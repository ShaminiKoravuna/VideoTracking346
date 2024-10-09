from __future__ import print_function

import numpy as np
import cv2

from time import time

from pyaer.davis import DAVIS

# Constant that contains the resolution of the DAVIS346
RES_W = 346  # Update this to the resolution of the DAVIS346
RES_H = 260

# Definition of the library class and its functions
class DVSAbsLib:

    # Constructor of the class with necessary parameters. 'pynq' is true when the library is executed on the Pynq
    def __init__(self, max_packet_size=None, max_packet_interval=None, hdmi_resolution=(640, 480), pynq=False, noise_filter=False):
        self.max_packet_size = max_packet_size
        self.max_packet_interval = max_packet_interval
        self.hdmi_resolution = hdmi_resolution
        self.clip_value = 0.5
        self.pynq = pynq
        self.noise_filter = noise_filter

    # Initializes all necessary objects
    def init(self):
        # If running on Pynq, import and initialize the overlay
        if self.pynq:
            #from pynq.overlays.base import BaseOverlay
            from pynq.overlays.base import BaseOverlay

            from pynq.lib.video import VideoMode

            # Load the Base overlay on the board
            base = BaseOverlay("base.bit")

            # Declare the object associated with HDMI output
            self.hdmi_out = base.video.hdmi_out

            # Configure HDMI output
            mode = VideoMode(self.hdmi_resolution[0], self.hdmi_resolution[1], 24)  # HDMI mode with resolution and bits per pixel
            self.hdmi_out.configure(mode)                                            # Apply mode and set pixels to grayscale
            self.hdmi_out.cacheable_frames = False                                   # Disable frame caching for faster IP movement
            self.hdmi_out.start()

        # Initialize the DAVIS camera
        self.device = DAVIS(noise_filter=self.noise_filter)

        # Start data stream with specified packet size and interval
        self.device.start_data_stream(max_packet_size=self.max_packet_size, max_packet_interval=self.max_packet_interval)

        if self.noise_filter:
            # Set bias from JSON configuration for noise filtering
            self.device.set_bias_from_json('configs/davis346_config.json')

    # Displays information about the connected DAVIS device
    def device_info(self):
        print("Device ID:", self.device.device_id)
        if self.device.device_is_master:
            print("Device is master.")
        else:
            print("Device is slave.")
        print("Device Serial Number:", self.device.device_serial_number)
        print("Device String:", self.device.device_string)
        print("Device USB bus Number:", self.device.device_usb_bus_number)
        print("Device USB device address:", self.device.device_usb_device_address)
        print("Device size X:", self.device.dvs_size_X)
        print("Device size Y:", self.device.dvs_size_Y)

    # Reads the event packet and returns the grayscale image
    def get_image(self, mode='hist'):
        if mode == 'hist':
            # Get event data
            event_data = self.device.get_event("events_hist")
            
            # Print out the returned data for debugging
            print("Returned event_data:", event_data)
            
            # Unpack the values based on observed structure
            pol_events, num_pol_event, special_events, num_special_event, _, _, _, _ = event_data

            if num_pol_event != 0:
                # Compute the captured image using positive and negative histograms
                if pol_events is not None:
                    img = pol_events[..., 1] - pol_events[..., 0]
                    img = np.clip(img, -self.clip_value, self.clip_value)  # Image in [-0.5, 0.5]
                    img = img + self.clip_value  # Image in [0, 1]
                    img = img * 255  # Image in grayscale [0, 255]
                    img = img.astype(np.uint8)
                    img = img.reshape((RES_H, RES_W))                         # Reshape to correct resolution
            
                else:
                    # If pol_events is None, handle appropriately
                    img = np.zeros((RES_H, RES_W), np.uint8)  # Default to an empty image

                return img, num_pol_event

            else:
                # Return a default image and zero count if no events
                img = np.zeros((RES_H, RES_W), np.uint8)  # Default to an empty image
                return img, 0

    # Handle other modes similarly...



        elif mode == 'event':
            # Returns an image where each pixel contains the number of mapped events
            (pol_events, num_pol_event, special_events, num_special_event) = self.device.get_event()

            if num_pol_event != 0:
                # Create an image of zeros
                img = np.zeros((RES_H, RES_W), np.uint32)

                for e in pol_events:
                    y = e[1]
                    x = e[2]

                    # Increment the event count at the pixel location
                    img[x, y] += 1

                return img, num_pol_event

        elif mode == 'time':
            # Returns a temporary image where each pixel contains the average timestamp of the mapped events
            (pol_events, num_pol_event, special_events, num_special_event) = self.device.get_event()

            if num_pol_event != 0:
                # Create arrays for event counts and timestamps
                events = np.zeros((RES_H, RES_W), np.uint32)
                time = np.zeros((RES_H, RES_W), np.uint32)

                t_0 = 0
                for e, i in zip(pol_events, range(0, num_pol_event)):
                    if i == 0:
                        t_0 = e[0]

                    t = e[0] - t_0
                    y = e[1]
                    x = e[2]

                    # Increment the event count and accumulate the timestamp
                    events[x, y] += 1
                    time[x, y] += t

                return np.divide(time, events), num_pol_event

        elif mode == 'raw':
            # Returns the raw event array and the number of events
            (pol_events, num_pol_event, special_events, num_special_event) = self.device.get_event()

            if num_pol_event != 0:
                return [pol_events, num_pol_event]

    # Displays the image via HDMI
    def hdmi_write(self, img: np.ndarray):
        if img.shape[0] < self.hdmi_resolution[0] and img.shape[1] < self.hdmi_resolution[1]:
            if self.pynq:
                # Display the frame via HDMI using the overlay
                outframe = self.hdmi_out.newframe()
                outframe[0:img.shape[0], 0:img.shape[1]] = img      # Fill only the part of the frame occupied by the DVS image
                self.hdmi_out.writeframe(outframe)
            else:
                print('*** ERROR: Pynq is not selected')
        else:
            print('*** ERROR: Input image size is greater than HDMI resolution')

    # Closes the HDMI interface to avoid issues when reprogramming the bitstream
    def hdmi_close(self):
        if self.pynq:
            self.hdmi_out.close()
        else:
            print('*** ERROR: Pynq is selected')

    # Displays the image using OpenCV, with optional scaling
    def show_img(self, img: np.ndarray, scale=1):
        if not self.pynq:
            if scale == 1:
                # Display the image directly
                cv2.imshow("Salida del proceso", img)
                cv2.waitKey(1)
            else:
                # Resize the image for better visibility
                res = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_AREA)
                cv2.imshow("Salida del proceso", res)
                cv2.waitKey(1)
        else:
            print('*** ERROR: Pynq is selected')
