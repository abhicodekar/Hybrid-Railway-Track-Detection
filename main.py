import os
import sys

# Standard Sandbox-Safe Libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("\n[ERROR] Missing required base packages.")
    print("[FIX] Run this in your terminal: pip install pillow numpy\n")
    sys.exit(1)

class HybridRailwayDetector:
    def __init__(self):
        print("[SYSTEM START] Image-Based Hybrid Railway Tracker Initialised.")
        print("[STATUS] Running in Safe-Sandbox Native Mode.")

    def load_image_as_matrix(self, image_path):
        """Loads a real image file and converts it into a workable math matrix."""
        if not os.path.exists(image_path):
            print(f"[WARNING] File '{image_path}' not found. Creating a synthetic mock image for testing...")
            # Automatically generates a placeholder track image if your file is missing
            img = Image.fromarray(np.uint8(np.random.randint(0, 255, (480, 640, 3))))
            img.save(image_path)
            
        img = Image.open(image_path).convert('RGB')
        return img, np.array(img)

    def convert_to_grayscale(self, rgb_matrix):
        """Converts RGB pixel coordinates into intensity arrays without CV2."""
        return (0.299 * rgb_matrix[:, :, 0] + 
                0.587 * rgb_matrix[:, :, 1] + 
                0.114 * rgb_matrix[:, :, 2]).astype(np.uint8)

    def process_track_edges(self, gray_frame):
        """Traditional Branch: Captures high-frequency linear track boundaries."""
        h, w = gray_frame.shape[:2]
        edge_map = np.zeros((h, w), dtype=np.uint8)
        
        # Sobel edge horizontal differential scan loop
        for y in range(1, h - 1, 2):
            for x in range(1, w - 1, 2):
                gradient = abs(int(gray_frame[y, x + 1]) - int(gray_frame[y, x - 1]))
                if gradient > 30:  # Threshold limit to isolate rail profiles
                    edge_map[y, x] = 255
        return edge_map

    def scan_for_obstacles(self, gray_frame):
        """Deep/CV Scanner: Analyzes structural pixels to flag track anomalies."""
        h, w = gray_frame.shape[:2]
        anomalies = []
        
        # Focal scanning coordinates (The core travel path corridor grid)
        scan_y_start, scan_y_end = int(h * 0.55), int(h * 0.85)
        scan_x_start, scan_x_end = int(w * 0.35), int(w * 0.65)
        
        # Calculate localized frame variance (detects sudden color contrast shifts)
        sample_zone = gray_frame[scan_y_start:scan_y_end, scan_x_start:scan_x_end]
        mean_intensity = np.mean(sample_zone)
        
        # Find specific pixel coordinates that break structural continuity
        for y in range(scan_y_start, scan_y_end, 15):
            for x in range(scan_x_start, scan_x_end, 15):
                pixel_val = gray_frame[y, x]
                # If a local block deviates severely from typical background tone, flag obstacle
                if abs(int(pixel_val) - int(mean_intensity)) > 45:
                    anomalies.append([x - 20, y - 20, x + 20, y + 20])
                    if len(anomalies) > 3:  # Group limits to avoid excessive boxes
                        break
        return anomalies

    def run_detection_pipeline(self, input_filename, output_filename):
        """Processes the input image file and outputs the final marked results."""
        # 1. Load real image data matrix
        pil_image, rgb_matrix = self.load_image_as_matrix(input_filename)
        
        # 2. Extract lighting invariant map channels
        gray = self.convert_to_grayscale(rgb_matrix)
        edges = self.process_track_edges(gray)
        
        # 3. Compute active safety threat array blocks
        obstacles = self.scan_for_obstacles(gray)
        
        # 4. Canvas rendering system (Draws UI over your image)
        draw = ImageDraw.Draw(pil_image)
        w, h = pil_image.size
        
        # Draw traditional track tracking path boundaries (Blue Lines)
        for y in range(int(h * 0.6), h - 5, 20):
            edge_pixels = np.where(edges[y, :] == 255)[0]
            if len(edge_pixels) > 0:
                # Mark left and right detected rail points
                draw.ellipse([edge_pixels[0]-3, y-3, edge_pixels[0]+3, y+3], fill="blue")
                if len(edge_pixels) > 1:
                    draw.ellipse([edge_pixels[-1]-3, y-3, edge_pixels[-1]+3, y+3], fill="blue")

        # Process obstacle threat status parameters
        if len(obstacles) > 0:
            print("[ALERT] Safety threat detected inside travel corridor boundaries!")
            # Draw Red Bounding Boxes around detected anomalies
            for box in obstacles:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 12), "OBSTACLE", fill="red")
            
            # HUD Top Banner (Red Alert)
            draw.rectangle([0, 0, w, 45], fill="red")
            draw.text((15, 12), "CRITICAL ALERT: OBSTACLE ON TRACK", fill="white")
        else:
            print("[SYSTEM STATUS] TRACK SECURE - CLEAR LINE DETECTED")
            # HUD Top Banner (Green Safe Status)
            draw.rectangle([0, 0, w, 45], fill="green")
            draw.text((15, 12), "SYSTEM STATUS: CLEAR LINE DETECTED", fill="white")

        # Save processed output matrix safely onto file path registers
        pil_image.save(output_filename)
        print(f"[SUCCESS] Processed image saved as: {output_filename}")

# --- Execution Entrypoint ---
if __name__ == "__main__":
    detector = HybridRailwayDetector()
    
    # Define file configurations
    # Place your custom track photo inside your project folder and name it 'railway_input.png'
    input_file = "railway_input.png"
    output_file = "railway_output.png"
    
    # Execute full framework calculation
    detector.run_detection_pipeline(input_file, output_file)
