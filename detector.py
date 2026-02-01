#!/usr/bin/env python3
"""
Word Problem Detection Demo

Point a camera at a piece of paper with a word problem on it. Detects when
it sees text that looks like a word problem, then captures a sharp image when
you press spacebar.

Features:
- Real-time detection with visual overlay
- Burst capture to get the sharpest frame
- Works with webcam and IP cameras
- Optional voice guidance to help you position the paper
"""

import cv2
import numpy as np
from collections import deque
import time
import os
from datetime import datetime
import platform
from enum import Enum


class State(Enum):
    """Tracks what the program is doing right now."""
    IDLE = "idle"          # showing camera feed
    SCANNING = "scanning"  # Looking for word problems
    CAPTURING = "capturing"  # Taking burst photo


class WordProblemDemo:
    def __init__(self, camera_url, output_dir="captures"):
        """Set up the detector."""
        # Convert string "0" to integer 0 for webcam
        if isinstance(camera_url, str) and camera_url.isdigit():
            self.camera_url = int(camera_url)
            self.is_webcam = True
        else:
            self.camera_url = camera_url
            self.is_webcam = False

        self.output_dir = output_dir
        self.cap = None
        self.running = True

        # Start in IDLE (not scanning)
        self.state = State.IDLE

        # Skip every other frame to keep video smooth
        self.frame_skip = 1
        self.frame_counter = 0

        # Remember last detection for skipped frames
        self.last_detected = False
        self.last_confidence = 0

        # Prevent accidental double-presses
        self.last_spacebar_time = 0
        self.spacebar_cooldown = 0.5  # 500ms minimum between presses

        # Keep last frame so video doesn't flicker
        self.cached_display = None

        # Auto-capture settings
        self.enable_auto_capture = False
        self.confidence_history = deque(maxlen=60)  # About 2 seconds at 30fps
        self.detection_history = deque(maxlen=60)   # Track if margins are OK
        self.auto_capture_threshold = 60  # Confidence needed
        self.auto_capture_duration = 2.0  # How long to wait
        self.auto_capture_triggered = False  # Don't trigger twice

        # Voice guidance (off by default)
        self.enable_voice_guidance = False
        self.guidance_cooldown = 3.0  # Wait between announcements
        self.last_guidance_time = 0

        os.makedirs(output_dir, exist_ok=True)
        print(f"Word Problem Demo initialized")
        print(f"Voice guidance: {'ENABLED' if self.enable_voice_guidance else 'DISABLED'}")

    def quick_precheck(self, frame):
        """
        Quick check to skip frames that obviously don't have text.
        Saves a lot of processing time.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Text needs decent lighting to be readable
        mean_brightness = np.mean(gray)
        if mean_brightness < 80 or mean_brightness > 200:
            return False

        # Text needs some variation (not just solid color)
        if gray.std() < 20:
            return False

        # Text has lots of edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # Should have 2-20% edge pixels for text
        if edge_density < 0.02 or edge_density > 0.25:
            return False

        return True

    def fast_rejection(self, frame):
        """
        Quick check to skip frames that are obviously bad.
        Returns: (should_reject: bool, reason: str)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Is the frame too blurry?
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return True, "Frame too blurry"

        # Is the lighting OK?
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            return True, "Too dark"
        if mean_brightness > 200:
            return True, "Too bright (overexposed)"

        # Is there enough contrast?
        contrast = np.std(gray)
        if contrast < 20:
            return True, "No contrast detected"

        return False, ""

    def extract_text_components(self, frame):
        """
        Find text in the frame. Returns a list of potential text pieces.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Make the text stand out from the background
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # Clean up small noise spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find all the blobs in the image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Figure out what size characters should be based on frame size
        min_char_height = int(h * 0.005)
        max_char_height = int(h * 0.15)
        min_char_width = int(w * 0.003)
        max_char_width = int(w * 0.20)
        min_area = min_char_height * min_char_width
        max_area = max_char_height * max_char_width * 3

        # Keep only blobs that look like text
        text_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Check if this blob looks like a character
            aspect_ratio = w_comp / h_comp if h_comp > 0 else 0

            if (min_area < area < max_area and
                min_char_height < h_comp < max_char_height and
                min_char_width < w_comp < max_char_width and
                0.05 < aspect_ratio < 5.0 and
                w_comp < w * 0.7 and
                h_comp < h * 0.4):

                text_components.append({
                    'x': x, 'y': y, 'w': w_comp, 'h': h_comp,
                    'area': area, 'aspect': aspect_ratio,
                    'center_x': x + w_comp/2,
                    'center_y': y + h_comp/2
                })

        # Remove lonely blobs that are probably noise
        if len(text_components) > 0:
            heights = [c['h'] for c in text_components]
            median_height = np.median(heights)
            text_components = self.filter_by_neighbor_density(text_components, median_height)

        return text_components

    def filter_by_neighbor_density(self, text_components, median_height):
        """
        Keep blobs that have neighbors nearby. Text has letters close together,
        random noise doesn't.
        """
        if len(text_components) == 0:
            return []

        search_radius = median_height * 2.5

        # Use a grid to make neighbor search faster
        grid = {}
        cell_size = search_radius

        # Put each component into a grid cell
        for i, comp in enumerate(text_components):
            cell_x = int(comp['center_x'] / cell_size)
            cell_y = int(comp['center_y'] / cell_size)
            key = (cell_x, cell_y)

            if key not in grid:
                grid[key] = []
            grid[key].append((i, comp))

        # Count neighbors for each component
        filtered_components = []

        for i, comp in enumerate(text_components):
            cell_x = int(comp['center_x'] / cell_size)
            cell_y = int(comp['center_y'] / cell_size)

            neighbor_count = 0

            # Check the 9 cells around this one
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (cell_x + dx, cell_y + dy)

                    if neighbor_key not in grid:
                        continue

                    for j, other in grid[neighbor_key]:
                        if i == j:
                            continue

                        # How far away is this neighbor?
                        dist_x = comp['center_x'] - other['center_x']
                        dist_y = comp['center_y'] - other['center_y']
                        distance = np.sqrt(dist_x*dist_x + dist_y*dist_y)

                        if distance < search_radius:
                            neighbor_count += 1

            # Keep it if it has at least 2 neighbors
            if neighbor_count >= 2:
                filtered_components.append(comp)

        return filtered_components

    def detect_lines(self, components, frame_height):
        """
        Group text components into lines.
        Returns: list of lines (each line is a list of components)
        """
        if len(components) == 0:
            return []

        # Sort top to bottom
        components.sort(key=lambda c: c['center_y'])

        # Figure out how far apart components can be to be on the same line
        median_height = np.median([c['h'] for c in components])
        line_tolerance = median_height * 0.6

        # Group components into lines
        lines = []
        current_line = []

        for comp in components:
            if not current_line:
                current_line.append(comp)
            else:
                prev_y = np.mean([c['center_y'] for c in current_line])
                if abs(comp['center_y'] - prev_y) < line_tolerance:
                    current_line.append(comp)
                else:
                    if len(current_line) >= 3:
                        lines.append(current_line)
                    current_line = [comp]

        if len(current_line) >= 3:
            lines.append(current_line)

        return lines

    def validate_margins(self, lines, frame_shape):
        """
        Check if the text is cut off at the edges.
        Returns: (margins_ok: bool, margin_info: dict)
        """
        h, w = frame_shape[:2]

        if len(lines) == 0:
            return False, {}

        # Get all the components from the lines
        line_components = []
        for line in lines:
            line_components.extend(line)

        if len(line_components) == 0:
            return False, {}

        # Find where the text starts and ends
        all_x = [c['x'] for c in line_components]
        all_y = [c['y'] for c in line_components]
        all_x2 = [c['x'] + c['w'] for c in line_components]
        all_y2 = [c['y'] + c['h'] for c in line_components]

        # Ignore a few outlier letters
        text_left = int(np.percentile(all_x, 2.5))
        text_top = int(np.percentile(all_y, 2.5))
        text_right = int(np.percentile(all_x2, 97.5))
        text_bottom = int(np.percentile(all_y2, 97.5))

        # Measure the margins
        left_margin = text_left
        top_margin = text_top
        right_margin = w - text_right
        bottom_margin = h - text_bottom

        text_width = text_right - text_left
        text_height = text_bottom - text_top

        # Need at least 2% margin on each side
        min_margin_x = w * 0.02
        min_margin_y = h * 0.02

        margins_ok = (left_margin > min_margin_x and
                     right_margin > min_margin_x and
                     top_margin > min_margin_y and
                     bottom_margin > min_margin_y)

        margin_info = {
            'text_left': text_left,
            'text_top': text_top,
            'text_right': text_right,
            'text_bottom': text_bottom,
            'text_width': text_width,
            'text_height': text_height,
            'left_margin': left_margin,
            'top_margin': top_margin,
            'right_margin': right_margin,
            'bottom_margin': bottom_margin,
            'margins_ok': margins_ok
        }

        return margins_ok, margin_info

    def validate_coverage(self, margin_info, frame_shape):
        """
        Check if the text is sized right in the frame.
        Returns: (coverage_ok: bool, coverage_info: dict)
        """
        h, w = frame_shape[:2]

        text_width = margin_info['text_width']
        text_height = margin_info['text_height']

        # What percentage of the frame does the text take up?
        width_ratio = text_width / w
        height_ratio = text_height / h
        area_ratio = (text_width * text_height) / (w * h)

        # Text should be big enough to read but not filling the whole frame
        width_ok = 0.25 < width_ratio < 0.95
        height_ok = 0.20 < height_ratio < 0.95
        area_ok = 0.10 < area_ratio < 0.85

        coverage_ok = width_ok and height_ok and area_ok

        coverage_info = {
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'area_ratio': area_ratio,
            'coverage_ok': coverage_ok
        }

        return coverage_ok, coverage_info

    def analyze_line_quality(self, lines):
        """
        Check how well-formed the lines are. Real text has consistent
        letter heights, straight lines, and even spacing.
        Returns: score from 0-50
        """
        if len(lines) == 0:
            return 0

        line_quality_scores = []

        for line in lines:
            heights = [c['h'] for c in line]
            mean_height = np.mean(heights)
            height_std = np.std(heights)

            height_consistency = height_std / mean_height if mean_height > 0 else 1.0

            y_positions = [c['center_y'] for c in line]
            y_std = np.std(y_positions)

            if len(line) >= 2:
                line.sort(key=lambda c: c['center_x'])
                gaps = [line[i+1]['center_x'] - line[i]['center_x'] for i in range(len(line)-1)]
                gap_std = np.std(gaps) if len(gaps) > 0 else 0
                mean_gap = np.mean(gaps) if len(gaps) > 0 else 0
                gap_consistency = gap_std / mean_gap if mean_gap > 5 else 1.0
            else:
                gap_consistency = 1.0

            line_score = 0
            if height_consistency < 0.3:  # Letters same height
                line_score += 33
            if y_std < 5:  # Letters on straight line
                line_score += 33
            if gap_consistency < 0.8:  # Even spacing
                line_score += 34

            line_quality_scores.append(line_score)

        avg_quality = np.mean(line_quality_scores)
        return (avg_quality / 100.0) * 50.0

    def calculate_confidence(self, components, lines, margin_info, coverage_info, quality_score):
        """
        Figure out how confident we are that this is a word problem.
        Returns: score from 0-100
        """
        confidence = 0.0

        # More text components = higher confidence (0-15 points)
        confidence += min(len(components) / 80.0, 1.0) * 15

        # More lines = higher confidence (0-20 points)
        confidence += min(len(lines) / 8.0, 1.0) * 20

        # How good do the lines look? (0-50 points)
        confidence += quality_score

        # Better margins = higher confidence (0-10 points)
        avg_margin = (margin_info['left_margin'] + margin_info['top_margin'] +
                     margin_info['right_margin'] + margin_info['bottom_margin']) / 4
        frame_diagonal = np.sqrt(margin_info['text_width']**2 + margin_info['text_height']**2)
        margin_quality = min(avg_margin / (frame_diagonal * 0.05), 1.0)
        confidence += margin_quality * 10

        # Text sized right in frame (0-5 points)
        area_ratio = coverage_info['area_ratio']
        coverage_score = 1.0 - abs(area_ratio - 0.5) / 0.5
        coverage_score = max(0, min(coverage_score, 1.0))
        confidence += coverage_score * 5

        return min(confidence, 100)

    def _scale_margin_info(self, margin_info, scale_factor):
        """
        We process frames at lower resolution for speed, but need to draw
        overlays at original resolution. This converts coordinates back.
        """
        if scale_factor == 1.0 or not margin_info:
            return margin_info

        # Scale all the coordinates
        scaled_info = margin_info.copy()
        scaled_info['text_left'] = int(margin_info['text_left'] / scale_factor)
        scaled_info['text_top'] = int(margin_info['text_top'] / scale_factor)
        scaled_info['text_right'] = int(margin_info['text_right'] / scale_factor)
        scaled_info['text_bottom'] = int(margin_info['text_bottom'] / scale_factor)
        scaled_info['text_width'] = int(margin_info['text_width'] / scale_factor)
        scaled_info['text_height'] = int(margin_info['text_height'] / scale_factor)
        scaled_info['left_margin'] = int(margin_info['left_margin'] / scale_factor)
        scaled_info['top_margin'] = int(margin_info['top_margin'] / scale_factor)
        scaled_info['right_margin'] = int(margin_info['right_margin'] / scale_factor)
        scaled_info['bottom_margin'] = int(margin_info['bottom_margin'] / scale_factor)

        return scaled_info

    def draw_detection_overlay(self, frame, components, lines, margin_info, confidence, detected):
        """
        Draw boxes and lines on the frame to show what we detected.
        """
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        if margin_info:
            text_left = margin_info['text_left']
            text_top = margin_info['text_top']
            text_right = margin_info['text_right']
            text_bottom = margin_info['text_bottom']
            margins_ok = margin_info['margins_ok']

            # Draw box around the text
            box_color = (0, 255, 0) if detected else (0, 200, 255)
            cv2.rectangle(display_frame, (text_left, text_top), (text_right, text_bottom),
                         box_color, 2)

            # Draw margin lines
            margin_color = (0, 255, 0) if margins_ok else (0, 100, 255)
            cv2.line(display_frame, (text_left, 0), (text_left, h), margin_color, 1)
            cv2.line(display_frame, (text_right, 0), (text_right, h), margin_color, 1)
            cv2.line(display_frame, (0, text_top), (w, text_top), margin_color, 1)
            cv2.line(display_frame, (0, text_bottom), (w, text_bottom), margin_color, 1)

        # Show debug info
        debug_text = f"Comps:{len(components)} Lines:{len(lines)} Conf:{confidence:.0f}%"
        cv2.putText(display_frame, debug_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display_frame

    def detect_word_problem(self, frame):
        """
        Main detection logic. Runs all the checks to see if we found a word problem.
        Returns: (detected, confidence, display_frame, positioning_data)
        """
        # Keep the original frame for drawing
        original_frame = frame
        h_orig, w_orig = frame.shape[:2]

        # Quick check to skip obviously bad frames
        if not self.quick_precheck(frame):
            positioning_data = {
                'has_text': False,
                'confidence': 0,
                'margins': None,
                'text_area_ratio': 0,
                'frame_shape': (h_orig, w_orig)
            }
            return False, 0, self._draw_status(original_frame.copy(), 0, "Searching for text"), positioning_data

        # Process at 640px wide for speed
        target_width = 640
        scale_factor = 1.0
        if w_orig > target_width:
            scale_factor = target_width / w_orig
            new_h = int(h_orig * scale_factor)
            frame = cv2.resize(original_frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame = original_frame

        h, w = frame.shape[:2]

        # Skip bad frames
        rejected, reason = self.fast_rejection(frame)
        if rejected:
            result = self._draw_status(original_frame.copy(), 0, reason)
            positioning_data = {
                'has_text': False,
                'confidence': 0,
                'margins': None,
                'text_area_ratio': 0,
                'frame_shape': (h_orig, w_orig)
            }
            return False, 0, result, positioning_data

        # Find text in the frame
        components = self.extract_text_components(frame)
        if len(components) < 10:
            result = self._draw_status(original_frame.copy(), 5, "Too few text components")
            positioning_data = {
                'has_text': False,
                'confidence': 5,
                'margins': None,
                'text_area_ratio': 0,
                'frame_shape': (h_orig, w_orig)
            }
            return False, 5, result, positioning_data

        # Group text into lines
        lines = self.detect_lines(components, h)
        if len(lines) < 2:
            result = self._draw_status(original_frame.copy(), 10, "Need multiple lines for word problem")
            positioning_data = {
                'has_text': len(components) >= 10,
                'confidence': 10,
                'margins': None,
                'text_area_ratio': 0,
                'frame_shape': (h_orig, w_orig)
            }
            return False, 10, result, positioning_data

        # Make sure text isn't cut off
        margins_ok, margin_info = self.validate_margins(lines, frame.shape)
        if not margins_ok:
            margin_info_scaled = self._scale_margin_info(margin_info, scale_factor)
            display_frame = self.draw_detection_overlay(original_frame, components, lines, margin_info_scaled, 15, False)
            result = self._draw_status(display_frame, 15, "Text cut off at edge")
            positioning_data = {
                'has_text': True,
                'confidence': 15,
                'margins': (margin_info['left_margin'],
                            margin_info['top_margin'],
                            margin_info['right_margin'],
                            margin_info['bottom_margin']),
                'text_area_ratio': (margin_info['text_width'] * margin_info['text_height']) / (w * h),
                'frame_shape': (h_orig, w_orig)
            }
            return False, 15, result, positioning_data

        # Make sure text is sized right
        coverage_ok, coverage_info = self.validate_coverage(margin_info, frame.shape)
        if not coverage_ok:
            margin_info_scaled = self._scale_margin_info(margin_info, scale_factor)
            display_frame = self.draw_detection_overlay(original_frame, components, lines, margin_info_scaled, 20, False)
            result = self._draw_status(display_frame, 20, "Problem not properly sized")
            positioning_data = {
                'has_text': True,
                'confidence': 20,
                'margins': (margin_info['left_margin'],
                            margin_info['top_margin'],
                            margin_info['right_margin'],
                            margin_info['bottom_margin']),
                'text_area_ratio': coverage_info['area_ratio'],
                'frame_shape': (h_orig, w_orig)
            }
            return False, 20, result, positioning_data

        # Check how good the lines look
        quality_score = self.analyze_line_quality(lines)

        # Calculate final confidence
        confidence = self.calculate_confidence(components, lines, margin_info,
                                              coverage_info, quality_score)

        # 60% confidence = good enough to capture
        detected = confidence >= 60

        # Convert coordinates back to original resolution
        margin_info_scaled = self._scale_margin_info(margin_info, scale_factor)

        # Draw boxes and lines on the frame
        display_frame = self.draw_detection_overlay(original_frame, components, lines,
                                                    margin_info_scaled, confidence, detected)

        # Show status message
        if detected:
            status = f"WORD PROBLEM ({confidence:.0f}%) - Press SPACE to capture"
        elif confidence >= 50:
            status = f"Almost there... ({confidence:.0f}%)"
        elif confidence >= 35:
            status = f"Positioning... ({confidence:.0f}%)"
        else:
            status = f"Searching for text ({confidence:.0f}%)"

        display_frame = self._draw_status(display_frame, confidence, status)

        # Save positioning info for voice guidance
        positioning_data = {
            'has_text': len(components) >= 10,
            'confidence': confidence,
            'margins': (margin_info['left_margin'],
                        margin_info['top_margin'],
                        margin_info['right_margin'],
                        margin_info['bottom_margin']) if margin_info else None,
            'text_area_ratio': coverage_info['area_ratio'] if coverage_info and 'area_ratio' in coverage_info else 0,
            'frame_shape': (h_orig, w_orig)
        }

        return detected, confidence, display_frame, positioning_data

    def analyze_positioning_issues(self, positioning_data):
        """
        Figure out what's wrong with the positioning so we can tell the user.
        """
        issues = {
            'cut_left': False,
            'cut_right': False,
            'cut_top': False,
            'cut_bottom': False,
            'too_close': False
        }

        if positioning_data.get('margins') is None:
            return issues

        margins = positioning_data['margins']
        h, w = positioning_data['frame_shape']

        # Need 2% margin on each edge
        threshold_x = w * 0.02
        threshold_y = h * 0.02

        # Check if text is too close to each edge
        if margins[0] < threshold_x:
            issues['cut_left'] = True
        if margins[2] < threshold_x:
            issues['cut_right'] = True
        if margins[1] < threshold_y:
            issues['cut_top'] = True
        if margins[3] < threshold_y:
            issues['cut_bottom'] = True

        # Check if camera is too close
        if positioning_data.get('text_area_ratio', 0) > 0.85:
            issues['too_close'] = True

        return issues

    def speak(self, text):
        """Text-to-speech to tell the user about positioning issues."""
        if not self.enable_voice_guidance:
            return

        # macOS has built-in speech
        if platform.system() == 'Darwin':
            os.system(f'say "{text}"')
        else:
            # Other platforms need pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except ImportError:
                print("Voice guidance requires pyttsx3: pip install pyttsx3")

    def provide_positioning_guidance(self, issues):
        """
        Tell the user how to fix their positioning.
        """
        if not self.enable_voice_guidance:
            return

        current_time = time.time()

        # Don't spam the user
        if current_time - self.last_guidance_time < self.guidance_cooldown:
            return

        # Any issues to report?
        has_issues = any(issues.values())
        if not has_issues:
            return

        # Build a message
        parts = []

        # Camera too close is most important
        if issues.get('too_close'):
            parts.append('Camera too close')

        # Which edges is the text cut off on?
        cut_sides = []
        if issues.get('cut_left'):
            cut_sides.append('left')
        if issues.get('cut_right'):
            cut_sides.append('right')
        if issues.get('cut_top'):
            cut_sides.append('top')
        if issues.get('cut_bottom'):
            cut_sides.append('bottom')

        if len(cut_sides) > 0:
            if len(cut_sides) == 1:
                parts.append(f"Text cut off on {cut_sides[0]}")
            elif len(cut_sides) == 2:
                parts.append(f"Text cut off on {cut_sides[0]} and {cut_sides[1]}")
            else:
                parts.append("Text cut off on multiple sides")

        if len(parts) > 0:
            message = ". ".join(parts)
            self.speak(message)
            self.last_guidance_time = current_time
            print(f"🗣️  Guidance: {message}")

    def _draw_status(self, frame, confidence, status_text):
        """Draw the status bar at the top of the frame."""
        h, w = frame.shape[:2]

        # Pick a color based on how confident we are
        if confidence >= 60:
            color = (0, 255, 0)
        elif confidence >= 50:
            color = (0, 255, 200)
        elif confidence >= 35:
            color = (0, 200, 255)
        else:
            color = (0, 100, 200)

        # Draw the status bar
        cv2.rectangle(frame, (10, 10), (w-10, 60), (0, 0, 0), -1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 60), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, status_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    def connect_camera(self):
        """Connect to the camera."""
        print(f"Connecting to camera: {self.camera_url}")

        if self.is_webcam:
            # macOS needs special backend
            if platform.system() == "Darwin":
                self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(self.camera_url)

            if not self.cap.isOpened():
                raise ConnectionError(f"Failed to connect to webcam (device {self.camera_url})")
        else:
            # IP cameras can have different URL patterns, try them all
            urls_to_try = [
                self.camera_url,
                f"{self.camera_url}/live/main",
                f"{self.camera_url}/stream",
                f"{self.camera_url}/main",
            ]

            for url in urls_to_try:
                self.cap = cv2.VideoCapture(url)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"✓ Successfully connected!")
                        self.camera_url = url
                        break
                self.cap.release()

            if not self.cap.isOpened():
                raise ConnectionError("Failed to connect to camera")

        print("Camera connected successfully")

    def capture_extended_burst(self, num_frames=120, duration=3.5):
        """
        Capture a bunch of frames over a few seconds. We'll pick the sharpest one later.
        """
        print(f"📸 Capturing {num_frames} frames over {duration:.1f}s...")

        frames = []
        start_time = time.time()
        target_interval = duration / num_frames

        while len(frames) < num_frames:
            elapsed = time.time() - start_time

            if elapsed > duration + 1.0:
                print(f"⏱️  Capture timeout at {elapsed:.1f}s")
                break

            ret, frame = self.cap.read()
            if ret and frame is not None:
                frames.append(frame.copy())

            time.sleep(max(0.005, target_interval * 0.7))

        actual_duration = time.time() - start_time
        actual_fps = len(frames) / actual_duration if actual_duration > 0 else 0

        print(f"✓ Captured {len(frames)} frames in {actual_duration:.2f}s (~{actual_fps:.1f} fps)")

        return frames

    def calculate_frame_sharpness(self, frame):
        """
        Calculate comprehensive sharpness score for frame.

        Returns: Float score (higher = sharper)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (primary sharpness indicator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()

        # Brightness score
        brightness = np.mean(gray)
        optimal_brightness = 140
        brightness_score = 1.0 - min(abs(brightness - optimal_brightness) / optimal_brightness, 1.0)

        # Contrast
        contrast = gray.std()
        contrast_score = min(contrast / 50.0, 1.0)

        # Combined score
        total_score = (
            laplacian_var * 0.70 +
            gradient_mag * 10 * 0.15 +
            brightness_score * 500 * 0.10 +
            contrast_score * 200 * 0.05
        )

        return total_score

    def select_best_frames(self, frames, top_n=1):
        """
        Select the sharpest frame(s) from burst.

        Args:
            frames: List of frames
            top_n: Number of top frames to return

        Returns:
            Single best frame (if top_n=1) or list of best frames
        """
        if len(frames) == 0:
            return None

        if len(frames) == 1:
            return frames[0]

        print(f"🔍 Scoring {len(frames)} frames for sharpness...")

        # Score all frames
        scored_frames = []
        for i, frame in enumerate(frames):
            score = self.calculate_frame_sharpness(frame)
            scored_frames.append((i, frame, score))

            if (i + 1) % 25 == 0:
                print(f"  ... scored {i + 1}/{len(frames)}")

        # Sort by score
        scored_frames.sort(key=lambda x: x[2], reverse=True)

        # Print top scores
        print(f"✓ Best frame scores:")
        for i in range(min(5, len(scored_frames))):
            idx, _, score = scored_frames[i]
            print(f"    Frame {idx}: {score:.1f}")

        if top_n == 1:
            best_idx, best_frame, best_score = scored_frames[0]
            print(f"✓ Selected frame {best_idx} (score: {best_score:.1f})")
            return best_frame
        else:
            top_frames = [frame for _, frame, _ in scored_frames[:top_n]]
            return top_frames

    def enhance_frame_minimal(self, frame):
        """
        Apply minimal enhancement with CLAHE.

        Args:
            frame: Input frame

        Returns:
            Enhanced frame
        """
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge back
        enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return result

    def save_capture(self, image):
        """Save captured image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"word_problem_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"✓ Saved: {filepath}")
        return filepath

    def _draw_idle_overlay(self, frame):
        """Draw idle state overlay (no detection)."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Gray status bar
        color = (150, 150, 150)
        cv2.rectangle(display, (10, 10), (w-10, 60), (0, 0, 0), -1)
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 60), color, -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.putText(display, "Press SPACE to start scanning", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display

    def _draw_capturing_overlay(self, frame):
        """Draw capturing state overlay with duration."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Blue status bar
        color = (255, 200, 0)
        cv2.rectangle(display, (10, 10), (w-10, 60), (0, 0, 0), -1)
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 60), color, -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # Show duration and instruction to hold still
        cv2.putText(display, "CAPTURING (3.5s)... HOLD STILL!", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display

    def check_auto_capture_ready(self):
        """Check if conditions met for auto-capture."""
        if not self.enable_auto_capture:
            return False

        # Need at least 2 seconds of history
        required_frames = int(self.auto_capture_duration * 30)  # Assume 30fps
        if len(self.confidence_history) < required_frames:
            return False

        # Check last N frames
        recent_confidence = list(self.confidence_history)[-required_frames:]
        recent_margins_ok = list(self.detection_history)[-required_frames:]

        # All frames must have confidence >= threshold
        if not all(c >= self.auto_capture_threshold for c in recent_confidence):
            return False

        # All frames must have margins OK (not cut off)
        if not all(m for m in recent_margins_ok):
            return False

        # Check stability (std dev < 5%)
        std_dev = np.std(recent_confidence)
        if std_dev > 5.0:
            return False

        return True

    def manual_capture(self):
        """Manual capture triggered by spacebar - synchronous execution."""
        self.state = State.CAPTURING

        # Capture burst
        frames = self.capture_extended_burst(num_frames=120, duration=3.5)

        if frames is None or len(frames) == 0:
            print("❌ Capture failed - no frames captured")
            self.state = State.IDLE
            return

        # Select best frame
        best_frame = self.select_best_frames(frames, top_n=1)

        if best_frame is None:
            print("❌ Frame selection failed")
            self.state = State.IDLE
            return

        # Enhance
        enhanced = self.enhance_frame_minimal(best_frame)

        # Save
        filepath = self.save_capture(enhanced)
        print(f"✅ Capture complete - Press SPACE to scan again")

        # Return to IDLE
        self.state = State.IDLE

    def run(self):
        """Main run loop."""
        try:
            self.connect_camera()

            print("\n" + "="*60)
            print("WORD PROBLEM DETECTION DEMO")
            print("="*60)
            print("Controls:")
            print("  SPACEBAR - Start scanning / Capture word problem")
            print("  Q        - Quit")
            print("\nFeatures:")
            print("  - Real-time detection with visual overlay")
            print("  - Burst capture for optimal sharpness")
            print(f"  - Voice guidance: {'ENABLED' if self.enable_voice_guidance else 'DISABLED'}")
            print("\nFlow:")
            print("  1. Press SPACE to start scanning")
            print("  2. Position paper in frame")
            print("  3. Press SPACE again to capture")
            print("="*60 + "\n")

            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    print("Failed to read frame, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.connect_camera()
                    continue

                self.frame_counter += 1

                # Handle state-specific display
                if self.state == State.IDLE:
                    # Show clean camera feed with instruction
                    display = self._draw_idle_overlay(frame)

                elif self.state == State.CAPTURING:
                    # Show capturing overlay
                    display = self._draw_capturing_overlay(frame)

                elif self.state == State.SCANNING:
                    # Run detection every 2nd frame
                    if self.frame_counter % (self.frame_skip + 1) == 0:
                        detected, confidence, display, positioning_data = self.detect_word_problem(frame)

                        # Cache complete display for smooth visuals
                        self.cached_display = display
                        self.last_confidence = confidence

                        # Track detection history for auto-capture
                        self.confidence_history.append(confidence)
                        margins_ok = positioning_data.get('margins', None) is not None
                        self.detection_history.append(margins_ok)

                        # Auto-capture check
                        if not self.auto_capture_triggered and self.check_auto_capture_ready():
                            print("🎯 Auto-capture triggered (sustained detection)!")
                            self.auto_capture_triggered = True
                            self.manual_capture()

                        # Voice guidance (if enabled)
                        if self.enable_voice_guidance and 15 <= confidence < 60:
                            issues = self.analyze_positioning_issues(positioning_data)
                            self.provide_positioning_guidance(issues)
                    else:
                        # Reuse cached display frame (no flickering!)
                        display = self.cached_display if self.cached_display is not None else self._draw_status(frame, 0, "Initializing...")

                # Display frame
                cv2.imshow("Word Problem Demo", display)

                # Handle keyboard with debouncing
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()

                if key == ord('q') or key == ord('Q'):
                    print("\nShutting down...")
                    self.running = False
                    break

                elif key == ord(' '):
                    # Debounce spacebar (ignore if pressed within cooldown)
                    if current_time - self.last_spacebar_time < self.spacebar_cooldown:
                        continue

                    self.last_spacebar_time = current_time

                    # State transition logic
                    if self.state == State.IDLE:
                        print("🔍 Starting scan...")
                        self.state = State.SCANNING
                        self.auto_capture_triggered = False  # Reset for new scan
                        self.confidence_history.clear()
                        self.detection_history.clear()

                    elif self.state == State.SCANNING:
                        print("📸 Capture triggered...")
                        self.manual_capture()  # This handles CAPTURING state internally

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Word Problem Detection Demo')
    parser.add_argument('camera_url', help='Camera URL or device index (e.g., 0 for webcam)')
    parser.add_argument('--auto-capture', action='store_true',
                       help='Enable automatic capture when word problem detected')
    args = parser.parse_args()

    demo = WordProblemDemo(args.camera_url)
    demo.enable_auto_capture = args.auto_capture

    if demo.enable_auto_capture:
        print("🤖 Auto-capture mode ENABLED (will capture automatically)")

    demo.run()


if __name__ == "__main__":
    main()
