#include <WiFi.h>
#include <WebServer.h>

// --------------------- AP Settings ---------------------
const char* ap_ssid = "ESP32_Sensor_AP"; // Name of the Wi‑Fi network
const char* ap_password = "12345678"; // Password (at least 8 characters)

// --------------------- Sensor Pins (ADC1) ---------------------
#define SOIL_PIN 32 // Capacitive soil sensor
#define HUMIDITY_PIN 33 // Analog humidity sensor

// --------------------- Calibration Constants ---------------------
// Replace these with your actual raw ADC readings:
const int soilDryValue = 3000; // ADC reading in dry air
const int soilWetValue = 1200; // ADC reading in water
const int humidDryValue = 0; // ADC reading at 0% humidity
const int humidWetValue = 4095; // ADC reading at 100% humidity

// --------------------- Averaging ---------------------
const int numReadings = 10;
int soilReadings[numReadings];
int humidReadings[numReadings];
int readIndex = 0;
long soilTotal = 0;
long humidTotal = 0;

// --------------------- Web Server ---------------------
WebServer server(80);

// Global variables to hold latest sensor values
int soilPercent = 0;
int humidPercent = 0;

// HTML page (inline for simplicity)
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<style>
body { font-family: Arial; text-align: center; margin-top: 50px; }
.value { font-size: 3em; color: #2c3e50; }
.label { font-size: 1.2em; color: #7f8c8d; }
</style>
</head>
<body>
<h1>ESP32 Sensor Data</h1>
<div>
<div class="label">Soil Moisture</div>
<div class="value">%SOIL% %</div>
</div>
<div style="margin-top:30px;">
<div class="label">Humidity</div>
<div class="value">%HUMID% %</div>
</div>
<p>Page auto-refreshes every 5 seconds.</p>
<p>JSON data available at <a href="/data.json">/data.json</a></p>
</body>
</html>
)rawliteral";

void setup() {
Serial.begin(115200);
delay(1000);

// Initialize averaging arrays
for (int i = 0; i < numReadings; i++) {
soilReadings[i] = 0;
humidReadings[i] = 0;
}

// Start Access Point
Serial.println("Starting Access Point...");
WiFi.softAP(ap_ssid, ap_password);
IPAddress IP = WiFi.softAPIP();
Serial.print("AP IP address: ");
Serial.println(IP);

// Set up web server routes
server.on("/", handleRoot);
server.on("/data.json", handleJSON);
server.begin();
Serial.println("HTTP server started");
}

void loop() {
// Read sensors (raw 0-4095)
int soilRaw = analogRead(SOIL_PIN);
int humidRaw = analogRead(HUMIDITY_PIN);

// Update moving averages
soilTotal = soilTotal - soilReadings[readIndex];
humidTotal = humidTotal - humidReadings[readIndex];

soilReadings[readIndex] = soilRaw;
humidReadings[readIndex] = humidRaw;

soilTotal = soilTotal + soilRaw;
humidTotal = humidTotal + humidRaw;

readIndex = (readIndex + 1) % numReadings;

int soilAvg = soilTotal / numReadings;
int humidAvg = humidTotal / numReadings;

// Convert to percentage (0-100)
soilPercent = map(soilAvg, soilDryValue, soilWetValue, 0, 100);
soilPercent = constrain(soilPercent, 0, 100);

humidPercent = map(humidAvg, humidDryValue, humidWetValue, 0, 100);
humidPercent = constrain(humidPercent, 0, 100);

// Print locally (optional)
Serial.print("Soil: "); Serial.print(soilPercent);
Serial.print("%, Humidity: "); Serial.print(humidPercent);
Serial.println("%");

// Handle client requests
server.handleClient();

delay(500); // Short delay, sensor updates frequently
}

// Serve the main HTML page
void handleRoot() {
String html = FPSTR(index_html);
html.replace("%SOIL%", String(soilPercent));
html.replace("%HUMID%", String(humidPercent));
server.send(200, "text/html", html);
}

// Serve JSON data
void handleJSON() {
String json = "{";
json += "\"soil\":" + String(soilPercent) + ",";
json += "\"humidity\":" + String(humidPercent);
json += "}";
server.send(200, "application/json", json);
}
