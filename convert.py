from fastkml import kml
from shapely.geometry import shape
import csv
import os
from xml.etree import ElementTree as ET

os.makedirs('data/ground_truth', exist_ok=True)

# Method 1: Try fastkml first
def extract_with_fastkml():
    with open('data/boundary/atewa.kml', 'rb') as f:
        k = kml.KML()
        k.from_string(f.read())
    
    features = []
    
    def extract_features(obj):
        """Recursively extract all Placemarks from KML object."""
        try:
            # Get features - handle both list and generator
            if hasattr(obj, 'features'):
                children = list(obj.features()) if callable(obj.features) else obj.features
                if children:
                    for child in children:
                        if hasattr(child, 'geometry') and child.geometry is not None:
                            features.append(child)
                        else:
                            extract_features(child)
        except Exception as e:
            print(f"Error in extract_features: {e}")
        return features
    
    return extract_features(k)

# Method 2: Direct XML parsing (more reliable)
def extract_with_xml():
    tree = ET.parse('data/boundary/atewa.kml')
    root = tree.getroot()
    
    # Register KML namespaces
    namespaces = {
        'kml': 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2'
    }
    
    placemarks = []
    
    # Find all Placemark elements
    for placemark in root.findall('.//kml:Placemark', namespaces):
        name_elem = placemark.find('kml:name', namespaces)
        name = name_elem.text if name_elem is not None else 'unnamed'
        
        # Find polygon coordinates
        coordinates_elem = placemark.find('.//kml:coordinates', namespaces)
        
        if coordinates_elem is not None and coordinates_elem.text:
            coords_text = coordinates_elem.text.strip()
            
            # Parse coordinates
            points = []
            for coord in coords_text.split():
                coord = coord.strip()
                if coord:
                    parts = coord.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        points.append((lon, lat))
            
            if points:
                # Create Shapely polygon
                from shapely.geometry import Polygon
                polygon = Polygon(points)
                
                # Simplify if needed (optional)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                
                placemarks.append({
                    'name': name,
                    'geometry': polygon,
                    'points': points
                })
    
    return placemarks

# Try fastkml first
print("Attempting to parse with fastkml...")
features = extract_with_fastkml()

if not features:
    print("fastkml found no features. Trying direct XML parsing...")
    features = extract_with_xml()

if features:
    print(f"Found {len(features)} placemarks")
    
    rows = []
    for feature in features:
        # Handle both fastkml objects and dict from XML parsing
        if isinstance(feature, dict):
            geom = feature['geometry']
            name = feature['name']
        else:
            geom = shape(feature.geometry)
            name = feature.name or 'digitized'
        
        centroid = geom.centroid
        # Calculate area in square meters (approximately)
        # Degrees to meters: ~111,319 meters per degree at equator
        area_m2 = int(geom.area * (111319 ** 2))
        area_m2 = max(area_m2, 100)  # Minimum 100 m2
        
        rows.append({
            'latitude': round(centroid.y, 6),
            'longitude': round(centroid.x, 6),
            'date': '2023-01-01',
            'site_type': 'galamsey',
            'active_status': 'active',
            'size_m2': area_m2,
            'notes': name,
        })
        print(f"  {name}: ({centroid.y:.6f}, {centroid.x:.6f})  area={area_m2}m2")
    
    # Save to CSV
    if rows:
        with open('data/ground_truth/mining_sites.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n✅ Saved {len(rows)} sites to data/ground_truth/mining_sites.csv')
    else:
        print("No rows generated")
else:
    print("❌ No features found. Checking KML file structure...")
    
    # Debug: Print KML structure
    tree = ET.parse('data/boundary/atewa.kml')
    root = tree.getroot()
    print("\nKML root tag:", root.tag)
    print("Number of children:", len(list(root)))
    
    # Find all Placemarks manually
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    placemarks = root.findall('.//kml:Placemark', namespaces)
    print(f"Placemarks found via XML: {len(placemarks)}")
    
    for pm in placemarks[:3]:  # Show first 3
        name = pm.find('kml:name', namespaces)
        print(f"  - {name.text if name is not None else 'unnamed'}")