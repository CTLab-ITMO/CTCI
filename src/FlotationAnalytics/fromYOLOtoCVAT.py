import os
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom

class DeepSortToCvatConverter:
    def __init__(self, input_folder, output_xml_file, img_width=None, img_height=None):
        self.input_folder = input_folder
        self.output_xml_file = output_xml_file
        self.img_width = img_width
        self.img_height = img_height

    def _create_metadata(self, annotations, num_frames):
        ET.SubElement(annotations, "version").text = "1.1"

        meta = ET.SubElement(annotations, "meta")
        job = ET.SubElement(meta, "job")

        ET.SubElement(job, "id").text = ""
        ET.SubElement(job, "size").text = str(num_frames)
        ET.SubElement(job, "mode").text = "annotation"
        ET.SubElement(job, "overlap").text = "0"
        ET.SubElement(job, "bugtracker")
        ET.SubElement(job, "created").text = "2025-04-18 13:40:56.502816+00:00"
        ET.SubElement(job, "updated").text = "2025-04-18 14:18:23.946959+00:00"
        ET.SubElement(job, "subset").text = "default"
        ET.SubElement(job, "start_frame").text = "0"
        ET.SubElement(job, "stop_frame").text = str(num_frames - 1)
        ET.SubElement(job, "frame_filter")

        segments = ET.SubElement(job, "segments")
        segment = ET.SubElement(segments, "segment")
        ET.SubElement(segment, "id").text = ""
        ET.SubElement(segment, "start").text = "0"
        ET.SubElement(segment, "stop").text = str(num_frames - 1)
        ET.SubElement(segment, "url").text = ""

        owner = ET.SubElement(job, "owner")
        ET.SubElement(owner, "username").text = ""
        ET.SubElement(owner, "email").text = ""

        ET.SubElement(job, "assignee")

        labels = ET.SubElement(job, "labels")
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = "bubble"
        ET.SubElement(label, "color").text = "#66ff66"
        ET.SubElement(label, "type").text = "any"
        ET.SubElement(label, "attributes")

        ET.SubElement(meta, "dumped").text = "2025-04-18 14:18:39.991110+00:00"

    def _process_tracks(self, annotations, txt_files):
        tracks = {}
        track_frames = {}
        active_tracks = set()

        for frame_num, filename in enumerate(txt_files):
            filepath = os.path.join(self.input_folder, filename)
            current_frame_tracks = set()

            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5:
                        continue

                    track_id = row[0]
                    current_frame_tracks.add(track_id)

                    x_center = float(row[1])
                    y_center = float(row[2])
                    width = float(row[3])
                    height = float(row[4])

                    if self.img_width and self.img_height and max(x_center, y_center, width, height) <= 1.0:
                        x_center *= self.img_width
                        y_center *= self.img_height
                        width *= self.img_width
                        height *= self.img_height

                    xtl = x_center
                    ytl = y_center
                    xbr = x_center + width
                    ybr = y_center + height

                    if self.img_width:
                        xtl = max(0, min(xtl, self.img_width))
                        xbr = max(0, min(xbr, self.img_width))
                    if self.img_height:
                        ytl = max(0, min(ytl, self.img_height))
                        ybr = max(0, min(ybr, self.img_height))

                    xtl, ytl, xbr, ybr = round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)

                    if track_id not in tracks:
                        track_elem = ET.SubElement(annotations, "track", {
                            "id": track_id,
                            "label": "bubble",
                            "source": "manual"
                        })
                        tracks[track_id] = track_elem
                        track_frames[track_id] = set()

                    if frame_num not in track_frames[track_id]:
                        ET.SubElement(tracks[track_id], "box", {
                            "frame": str(frame_num),
                            "keyframe": "1",
                            "outside": "0",
                            "occluded": "0",
                            "xtl": str(xtl),
                            "ytl": str(ytl),
                            "xbr": str(xbr),
                            "ybr": str(ybr),
                            "z_order": "0"
                        })
                        track_frames[track_id].add(frame_num)

            disappeared_tracks = active_tracks - current_frame_tracks
            for track_id in disappeared_tracks:
                if track_id in tracks:
                    ET.SubElement(tracks[track_id], "box", {
                        "frame": str(frame_num),
                        "keyframe": "1",
                        "outside": "1",
                        "occluded": "0",
                        "xtl": "0",
                        "ytl": "0",
                        "xbr": "0",
                        "ybr": "0",
                        "z_order": "0"
                    })
                    track_frames[track_id].add(frame_num)

            active_tracks = current_frame_tracks

    def convert(self):
        txt_files = sorted(f for f in os.listdir(self.input_folder) if f.endswith('.txt'))
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.input_folder}")

        annotations = ET.Element("annotations")
        
        self._create_metadata(annotations, len(txt_files))
        
        self._process_tracks(annotations, txt_files)

        xml_str = ET.tostring(annotations, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        with open(self.output_xml_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        print(f"Successfully converted {len(txt_files)} frames to CVAT XML at {self.output_xml_file}")
