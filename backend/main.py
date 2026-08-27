from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- SETUP & CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 314

CLASS_NAMES = ['airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation', 'archive', 'army_base', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'asia', 'assembly_line', 'attic', 'auditorium', 'auto_factory', 'auto_showroom', 'badlands', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_vault', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball', 'baseball_field', 'basement', 'bathroom', 'beach', 'beach_house', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth', 'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'botanical_garden', 'bowling_alley', 'boxing_ring', 'bridge', 'broadleaf', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cafeteria', 'campsite', 'campus', 'candy_store', 'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb', 'cemetery', 'chalet', 'chemistry_lab', 'childs_room', 'classroom', 'clean_room', 'cliff', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'corn_field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'cultivated', 'dam', 'delicatessen', 'department_store', 'desert_road', 'dining_hall', 'dining_room', 'discotheque', 'door', 'dorm_room', 'downtown', 'dressing_room', 'driveway', 'drugstore', 'elevator_lobby', 'elevator_shaft', 'embassy', 'engine_room', 'entrance_hall', 'excavation', 'exterior', 'fabric_store', 'farm', 'fastfood_restaurant', 'field_road', 'fire_escape', 'fire_station', 'fishpond', 'food_court', 'football', 'football_field', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'gas_station', 'gift_shop', 'glacier', 'golf_course', 'grotto', 'harbor', 'hardware_store', 'hayfield', 'heliport', 'highway', 'hockey', 'home_office', 'home_theater', 'hospital', 'hospital_room', 'hot_spring', 'hotel_room', 'house', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'iceberg', 'igloo', 'indoor', 'industrial_area', 'interior', 'islet', 'jail_cell', 'japanese_garden', 'jewelry_shop', 'junkyard', 'kasbah', 'kindergarden_classroom', 'kitchen', 'lagoon', 'landfill', 'landing_deck', 'laundromat', 'lawn', 'lecture_room', 'legislative_chamber', 'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine', 'motel', 'mountain', 'mountain_path', 'mountain_snowy', 'music_studio', 'natural', 'natural_history_museum', 'nursery', 'nursing_home', 'oast_house', 'ocean', 'ocean_deep', 'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room', 'orchard', 'orchestra_pit', 'outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_lot', 'pasture', 'patio', 'pavilion', 'performance', 'pet_shop', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'platform', 'playground', 'playroom', 'plaza', 'pond', 'porch', 'promenade', 'public', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'repair_shop', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'rodeo', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sand', 'sandbox', 'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shed', 'shoe_shop', 'shop', 'shopfront', 'shower', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'soccer', 'soccer_field', 'stable', 'staircase', 'storage_room', 'street', 'supermarket', 'sushi_bar', 'swamp', 'swimming_hole', 'television_room', 'television_studio', 'throne_room', 'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior', 'tree_farm', 'tree_house', 'trench', 'tundra', 'urban', 'utility_room', 'valley', 'vegetable_garden', 'vegetation', 'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano', 'waiting_room', 'water', 'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wild', 'wind_farm', 'windmill', 'yard', 'youth_hostel', 'zen_garden']

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODEL DEFINITIONS ---
class MobileNetSceneClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None) 
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class SceneClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# --- MODEL LOADING ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load MobileNet
    ml_models["mobilenet"] = MobileNetSceneClassifier(num_classes=NUM_CLASSES).to(device)
    ml_models["mobilenet"].load_state_dict(torch.load('mobilenet_scene_classifier-3.pth', map_location=device))
    ml_models["mobilenet"].eval()

    # Load DINOv2
    ml_models["dinov2"] = SceneClassifier(num_classes=NUM_CLASSES).to(device)
    ml_models["dinov2"].load_state_dict(torch.load('best_scene_classifier.pth', map_location=device))
    ml_models["dinov2"].eval()
    
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Allow React to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PREDICTION ENDPOINT (TOP 5) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = eval_transform(image).unsqueeze(0).to(device)
    
    results = {}
    
    with torch.no_grad():
        # --- Run MobileNet ---
        mn_outputs = ml_models["mobilenet"](image_tensor)
        mn_prob = torch.nn.functional.softmax(mn_outputs, dim=1)
        mn_top5_prob, mn_top5_idx = torch.topk(mn_prob, 5)
        
        mobilenet_preds = []
        for i in range(5):
            mobilenet_preds.append({
                "className": CLASS_NAMES[mn_top5_idx[0][i].item()],
                "confidence": round(mn_top5_prob[0][i].item() * 100, 2)
            })
            
        results["mobilenet"] = mobilenet_preds

        # --- Run DINOv2 ---
        dino_outputs = ml_models["dinov2"](image_tensor)
        dino_prob = torch.nn.functional.softmax(dino_outputs, dim=1)
        dino_top5_prob, dino_top5_idx = torch.topk(dino_prob, 5)

        dinov2_preds = []
        for i in range(5):
            dinov2_preds.append({
                "className": CLASS_NAMES[dino_top5_idx[0][i].item()],
                "confidence": round(dino_top5_prob[0][i].item() * 100, 2)
            })
            
        results["dinov2"] = dinov2_preds
        
    return results