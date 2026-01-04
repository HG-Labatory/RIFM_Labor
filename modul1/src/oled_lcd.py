import subprocess
import time
import math
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import ImageFont, ImageDraw
from influxdb_client import InfluxDBClient
import random



# I2C-Verbindung zum OLED-Display herstellen (Ändere "sh1106" auf "ssd1306", falls dein Display den SSD1306-Chip hat)
serial = i2c(port=1, address=0x3C)  # Ändere auf 0x3D, falls dein Display eine andere Adresse hat
device = sh1106(serial)


# Schriftart laden
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 8)

# InfluxDB Verbindung
INFLUXDB_URL = "http://localhost:8086"  # Falls dein Influx woanders läuft, anpassen
INFLUXDB_TOKEN = None  # Falls InfluxDB 2.x, sonst None lassen
INFLUXDB_ORG = None  # Falls InfluxDB 2.x, sonst None lassen
INFLUXDB_BUCKET = "sensor_db"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# 3D-Würfel Koordinaten (8 Punkte)
CUBE_POINTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
]

# Kanten des Würfels (Verbindungen der Punkte)
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Vorderseite
    (4, 5), (5, 6), (6, 7), (7, 4),  # Rückseite
    (0, 4), (1, 5), (2, 6), (3, 7)   # Verbindungen
]



# Temperatur auslesen (Raspberry Pi)
def get_cpu_temperature():
    """ Liest die CPU-Temperatur aus dem Raspberry Pi aus """
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp = float(output.replace("temp=", "").replace("'C\n", ""))
        return temp
    except Exception:
        return "N/A"
    
# Temperatur auslesen (Raspberry Pi)
def get_voltage():
    """ Liest die CPU-Temperatur aus dem Raspberry Pi aus """
    try:
        output = subprocess.check_output(["vcgencmd", "measure_volts"]).decode()
        volt = float(output.replace("volt=", "").replace("V\n", ""))
        return volt
    except Exception:
        return "N/A"

# Funktion zum Abrufen eines Werts aus InfluxDB
def get_sensor_value(measurement, field="usage_idle"):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> filter(fn: (r) => r._field == "{field}")
        |> last()
    '''
    result = query_api.query(query)

    for table in result:
        for record in table.records:
            return round(record.get_value(), 2)  # Werte runden

    return "N/A"


def check_service(service_name):
    result = subprocess.run(["systemctl", "is-active", service_name], capture_output=True, text=True)
    return "OK" if result.stdout.strip() == "active" else "X"


def animate_double_pyramid():
    """ Zeichnet eine rotierende Doppelpyramide (Oktaeder) auf dem OLED-Display """

    cx, cy = 80, 32  # Mittelpunkt des Displays
    size = 0.6  # Größe der Pyramide

    # 3D-Punkte des Oktaeders (Doppelpyramide)
    vertices = [
        (0, -1, 0),  # Spitze oben
        (-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1),  # Mittlere Punkte (Basis)
        (0, 1, 0)  # Spitze unten
    ]

    # Kanten der Doppelpyramide (Verbindungen zwischen den Punkten)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Verbindung von oberer Spitze zur Basis
        (5, 1), (5, 2), (5, 3), (5, 4),  # Verbindung von unterer Spitze zur Basis
        (1, 2), (2, 3), (3, 4), (4, 1)   # Basisverbindungen
    ]

    angle_x, angle_y = 0, 0  # Startwinkel für Rotation

    for _ in range(200):  # 15 Sekunden Animation (30 Frames, 0.5s pro Frame)
        cpu_temp = get_cpu_temperature() 
        voltage = round(get_voltage(),2)  
        with canvas(device) as draw:
            draw.text((0, 5),  f"{cpu_temp} C", font=font, fill="white")
            draw.text((0, 15),  f"{voltage} V", font=font, fill="white")
            transformed_points = []

            # 3D-Transformation für Drehung
            for x, y, z in vertices:
                # Rotation um die X-Achse
                y2 = y * math.cos(angle_x) - z * math.sin(angle_x)
                z2 = y * math.sin(angle_x) + z * math.cos(angle_x)

                # Rotation um die Y-Achse
                x2 = x * math.cos(angle_y) - z2 * math.sin(angle_y)
                z3 = x * math.sin(angle_y) + z2 * math.cos(angle_y)

                # 3D-zu-2D-Projektion
                scale = 100 / (z3 + 3)
                x2d = int(cx + x2 * size * scale)
                y2d = int(cy + y2 * size * scale)

                transformed_points.append((x2d, y2d))

            # Linien für die Doppelpyramide zeichnen
            for edge in edges:
                draw.line([transformed_points[edge[0]], transformed_points[edge[1]]], fill="white")

        # Rotation erhöhen
        angle_x += math.pi / 20  # Rotation um X-Achse
        angle_y += math.pi / 25  # Rotation um Y-Achse
        time.sleep(0.05)  # Flüssige Animation

def animate_sphere():
    """ Zeichnet eine rotierende 3D-Kugel auf dem OLED-Display """

    cx, cy = 70, 40  # Mittelpunkt des Displays
    radius = 9  # Größe der Kugel
    light_source = (1, -1, 1)  # Lichtquelle von oben links

    for angle_y in range(0, 360, 5):  # Rotation in 10°-Schritten
        cpu_temp = get_cpu_temperature() 
        voltage = round(get_voltage(),2)  
        with canvas(device) as draw:
            draw.text((0, 5),  f"{cpu_temp} C", font=font, fill="white")
            draw.text((0, 15),  f"{voltage} V", font=font, fill="white")
            for theta in range(0, 360, 10):  # Breitengrade
                for phi in range(0, 180, 10):  # Längengrade (halbe Kugel)

                    # Kugelkoordinaten berechnen
                    x = radius * math.sin(math.radians(phi)) * math.cos(math.radians(theta))
                    y = radius * math.sin(math.radians(phi)) * math.sin(math.radians(theta))
                    z = radius * math.cos(math.radians(phi))

                    # Rotation um die Y-Achse
                    x_rot = x * math.cos(math.radians(angle_y)) - z * math.sin(math.radians(angle_y))
                    z_rot = x * math.sin(math.radians(angle_y)) + z * math.cos(math.radians(angle_y))

                    # 3D-zu-2D-Projektion
                    scale = 100 / (z_rot + radius + 20)  # Perspektive verstärken
                    projx = int(cx + x_rot * scale)
                    projy = int(cy + y * scale)

                    # Beleuchtungseffekt basierend auf dem Lichtwinkel
                    dot_product = (x_rot * light_source[0] + y * light_source[1] + z_rot * light_source[2])
                    brightness = max(0, dot_product / radius)  # Helligkeit berechnen

                    # Kugel zeichnen (hellere Punkte für Licht, dunklere für Schatten)
                    color = int(255 * brightness)  # Skaliere Helligkeit
                    if 0 <= projx < 128 and 0 <= projy < 64:
                        draw.point((projx, projy), fill="white" if color > 128 else "black")

        time.sleep(0.05)  # Flüssige Bewegung

# Funktion zum Zeichnen des 3D-Würfels mit Drehung
def draw_cube(draw, angle_x, angle_y):
    """ Zeichnet einen rotierenden 3D-Würfel auf dem OLED-Display """

    cx, cy = 80, 32  # Mittelpunkt des Displays für SSD1306 (128x64)
    size = 0.6  # Würfelgröße für größere Darstellung
    points = []

    # 3D-Punkte berechnen
    for x, y, z in CUBE_POINTS:
        # Rotation um die X-Achse
        y2 = y * math.cos(angle_x) - z * math.sin(angle_x)
        z2 = y * math.sin(angle_x) + z * math.cos(angle_x)

        # Rotation um die Y-Achse
        x3 = x * math.cos(angle_y) - z2 * math.sin(angle_y)
        z3 = x * math.sin(angle_y) + z2 * math.cos(angle_y)

        # 3D-zu-2D-Projektion
        scale = 80 / (z3 + 3)  # Größere Skalierung für SSD1306
        x2d = int(cx + x3 * size * scale)
        y2d = int(cy + y2 * size * scale)

        points.append((x2d, y2d))

    # Linien für den Würfel zeichnen
    for edge in CUBE_EDGES:
        draw.line([points[edge[0]], points[edge[1]]], fill="white")

# Funktion zur Darstellung des rotierenden Würfels
def animate_cube():
    angle_x, angle_y = 0, 0  # Startwinkel
    for _ in range(200):  # 15 Sekunden Animation (30 Frames, 0.5s pro Frame)
        cpu_temp = get_cpu_temperature() 
        voltage = round(get_voltage(),2)      
        with canvas(device) as draw:
            draw.text((0, 5),  f"{cpu_temp} C", font=font, fill="white")
            draw.text((0, 15),  f"{voltage} V", font=font, fill="white")
            draw_cube(draw, angle_x, angle_y)  # Würfel mit aktuellem Winkel zeichnen
        angle_x += math.pi / 20  # Drehung um X-Achse
        angle_y += math.pi / 25  # Drehung um Y-Achse
        time.sleep(0.05)  # Wartezeit zwischen Frames
 
# Sinuswellen-Animation
def animate_sine_wave():
    """ Zeigt eine rotierende Sinuswelle auf dem OLED an """
    for frame in range(300):  # 15 Sekunden Animation (30 Frames, 0.5s pro Frame)
        with canvas(device) as draw:
            for x in range(0, 128, 4):  # Alle 4 Pixel eine Linie zeichnen
                y = int(32 + 15 * math.sin((x + frame * 4) * math.pi / 64))  # Sinuswelle
                draw.line([(x, 32), (x, y)], fill="white")  # Vertikale Linien erzeugen
        time.sleep(0.03)  # Wartezeit für Animation  
        
# Funktion zur Aktualisierung des OLED-Displays
def update_display():
    
    while True:

        for i in range(5):  # 15 Sekunden lang anzeigen
            
            # Status der Dienste abfragen
            status_mqtt = check_service("mosquitto")
            status_telegraf = check_service("telegraf")
            status_influxdb = check_service("influxdb")
            status_grafana = check_service("grafana-server")
            
            # Status und CPU-Temperatur auf OLED anzeigen
            with canvas(device) as draw:
                draw.text((0, 5),  f"MQTT      : {status_mqtt}", font=font, fill="white")
                draw.text((0, 15), f"Telegraf  : {status_telegraf}", font=font, fill="white")
                draw.text((0, 25), f"InfluxDB  : {status_influxdb}", font=font, fill="white")
                draw.text((0, 35), f"Grafana   : {status_grafana}", font=font, fill="white")
            
            time.sleep(1)
            
        for i in range(5):  # 15 Sekunden lang anzeigen
            cpu_usage = round(100 - get_sensor_value("cpu", "usage_idle"), 2)
            mem_usage = round(get_sensor_value("mem", "used_percent"), 2)
            disk_usage = round(get_sensor_value("disk", "used_percent"), 2)
            system_uptime = round(get_sensor_value("system", "uptime"), 0)

            uptime_hours = int(system_uptime // 3600)
            uptime_minutes = int((system_uptime % 3600) // 60)
        
            # OLED-Display aktualisieren
            with canvas(device) as draw:
                draw.text((0, 5),  f"CPU    :  {cpu_usage} %", font=font, fill="white")
                draw.text((0, 15), f"RAM    :  {mem_usage} %", font=font, fill="white")
                draw.text((0, 25), f"Disk   : {disk_usage} %", font=font, fill="white")
                draw.text((0, 35), f"Uptime : {uptime_hours} h {uptime_minutes} m", font=font, fill="white")

            time.sleep(1)
            
        
        animate_cube()  # Starte die Animation!
        animate_double_pyramid()
        animate_sphere()
            
# Funktion starten
update_display()