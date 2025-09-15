PRODUCT_CATALOG = [
    # Gaming and Electronics
    "Gaming laptop with RTX 4070 GPU, 16GB RAM, 1TB SSD, 144Hz display",
    "High-performance gaming desktop with RTX 4080, AMD Ryzen 7, RGB lighting",
    "Gaming mechanical keyboard with cherry MX switches, RGB backlight, programmable macros",
    "Wireless gaming mouse with 25600 DPI, programmable buttons, RGB lighting",
    "Gaming headset with 7.1 surround sound, noise cancellation, retractable microphone",
    "34-inch curved gaming monitor, 144Hz, QHD resolution, G-Sync compatible",
    "Gaming chair with lumbar support, adjustable armrests, premium leather padding",
    
    # Laptops and Computers
    "Business laptop with Intel i7, 32GB RAM, 512GB SSD, 14-inch display",
    "Ultrabook with long battery life, lightweight design, touchscreen display",
    "2-in-1 convertible laptop with stylus support, detachable keyboard",
    "Workstation laptop for video editing, Quadro GPU, color-accurate display",
    "Budget laptop for students, AMD processor, 8GB RAM, good battery life",
    "MacBook Pro alternative with ARM processor, excellent build quality",
    
    # Audio Equipment
    "Wireless noise-cancelling headphones with 30-hour battery, premium sound quality",
    "Professional studio headphones for music production, flat response curve",
    "Bluetooth earbuds with active noise cancellation, wireless charging case",
    "Noise-isolating in-ear monitors for musicians, custom-molded fit available",
    "Portable Bluetooth speaker, waterproof, 360-degree sound, long battery",
    "Smart speaker with voice assistant, multi-room audio, premium sound",
    "Soundbar with subwoofer, Dolby Atmos, wireless connectivity",
    
    # Mobile Devices
    "Flagship smartphone with 128GB storage, 5G support, triple camera system",
    "Budget smartphone with good camera, all-day battery, clean Android",
    "Rugged smartphone for outdoor use, waterproof, drop-resistant, long battery",
    "Smartphone with exceptional camera for photography enthusiasts",
    "Foldable smartphone with innovative design, premium build quality",
    "Smartwatch with health tracking, GPS, cellular connectivity, premium materials",
    "Fitness tracker with heart rate monitor, sleep tracking, water resistant",
    
    # Home and Office
    "Ergonomic office chair with lumbar support, adjustable height, breathable mesh",
    "Adjustable standing desk with memory presets, cable management, sturdy build",
    "LED desk lamp with wireless charging pad, adjustable brightness, USB ports",
    "Air purifier with HEPA filter, smart sensors, quiet operation, large room coverage",
    "Robot vacuum cleaner with smart mapping, automatic emptying, pet hair removal",
    "Smart thermostat with energy saver mode, learning algorithms, remote control",
    "Home security camera system with night vision, motion detection, cloud storage",
    
    # Appliances and Kitchen
    "Espresso machine with milk frother, programmable settings, stainless steel",
    "Air fryer with 6-quart capacity, digital display, multiple cooking presets",
    "Stainless steel cookware set, 10 pieces, non-stick coating, dishwasher safe",
    "High-power blender for smoothies, ice crushing, multiple speed settings",
    "Smart refrigerator with touchscreen, internal cameras, energy efficient",
    "Induction cooktop with precise temperature control, safety features",
    
    # Outdoor and Sports
    "Electric mountain bike with 500W motor, long-range battery, suspension",
    "Road bike with carbon fiber frame, lightweight, professional components",
    "Camping tent for 4 people, waterproof, easy setup, compact when packed",
    "Hiking backpack with hydration system, multiple compartments, weather resistant",
    "Adjustable dumbbell set, 5-50 lbs, space-saving design, quick adjustment",
    "Yoga mat with non-slip surface, 8mm thick, eco-friendly materials",
    "Running shoes with responsive cushioning, breathable mesh, lightweight design",
    
    # Health and Personal Care
    "Electric toothbrush with pressure sensor, multiple brush modes, long battery",
    "Premium mattress with cooling gel memory foam, motion isolation, 10-year warranty",
    "Massage chair with full body coverage, multiple massage techniques, heat therapy",
    "Smart scale with body composition analysis, app connectivity, multiple users",
    "Air quality monitor with real-time data, mobile app, multiple sensor types",
    
    # Storage and Accessories
    "External SSD, 2TB, USB-C, portable design, fast data transfer speeds",
    "Wireless charging pad for multiple devices, fast charging, sleek design",
    "Power bank, 20,000mAh capacity, fast charging, multiple device support",
    "USB hub with multiple ports, compact design, high-speed data transfer",
    "Cloud storage backup device for automatic file synchronization",
    
    # Entertainment and Media
    "4K Ultra HD Smart TV, 65 inch, HDR support, streaming apps, voice control",
    "High-resolution DSLR camera with 24MP sensor, interchangeable lenses",
    "Action camera with 4K recording, image stabilization, waterproof housing",
    "Streaming device with 4K support, voice remote, extensive app library",
    "Gaming console with exclusive titles, backward compatibility, 4K gaming",
    
    # Tools and Equipment
    "Cordless drill set with multiple bits, fast charging, LED work light",
    "Laser measuring tool with high accuracy, digital display, compact design",
    "Multi-tool with various functions, durable construction, lifetime warranty",
    "Pressure washer for outdoor cleaning, adjustable pressure, multiple nozzles",
    "Cordless stick vacuum with HEPA filter, lightweight, versatile attachments",
    
    # Fashion and Lifestyle
    "Winter jacket with thermal insulation, waterproof, breathable fabric",
    "Leather messenger bag with laptop compartment, durable construction",
    "Premium sunglasses with UV protection, polarized lenses, stylish design",
    "Cotton crew neck T-shirt collection, various colors, comfortable fit",
    "Stainless steel water bottle, 1L capacity, insulated, leak-proof design",
    
    # Smart Home and IoT
    "Smart home hub with voice assistant, device control, automation features",
    "Smart door lock with keypad entry, smartphone control, security features",
    "Smart light bulbs with color changing, dimming, voice control, energy efficient",
    "Security system with cameras, sensors, smartphone alerts, professional monitoring",
    "Smart garage door opener with smartphone control, security features"
]

# Product categories for better organization
CATEGORIES = {
    "gaming": ["gaming", "rgb", "mechanical", "fps", "performance"],
    "laptop": ["laptop", "notebook", "portable", "mobile", "ultrabook"],
    "audio": ["headphones", "speaker", "sound", "music", "audio"],
    "smartphone": ["phone", "mobile", "cellular", "5g", "camera"],
    "home": ["chair", "desk", "office", "home", "furniture"],
    "kitchen": ["cooking", "kitchen", "food", "appliance"],
    "fitness": ["exercise", "workout", "sports", "running", "fitness"],
    "outdoor": ["camping", "hiking", "outdoor", "bike", "adventure"]
}

def get_products_by_category(category: str, products: list = None):
    """Filter products by category keywords."""
    if products is None:
        products = PRODUCT_CATALOG
    
    if category not in CATEGORIES:
        return []
    
    keywords = CATEGORIES[category]
    filtered = []
    
    for product in products:
        product_lower = product.lower()
        if any(keyword in product_lower for keyword in keywords):
            filtered.append(product)
    
    return filtered

def search_products(query: str, products: list = None):
    """Simple keyword search in products."""
    if products is None:
        products = PRODUCT_CATALOG
    
    query_lower = query.lower()
    results = []
    
    for product in products:
        if query_lower in product.lower():
            results.append(product)
    
    return results