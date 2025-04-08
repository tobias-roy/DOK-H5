#include <WiFiNINA.h>
#include "arduino_secrets.h"
#include <utility/wifi_drv.h>
#include <utility/ECCX08SelfSignedCert.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include <Servo.h>
#include <ArduinoMqttClient.h>
#include <ArduinoJson.h>
#include <ArduinoECCX08.h>
#include <ArduinoBearSSL.h>

#define DHTPIN A3
#define SERVOPIN A4
#define DHTTYPE DHT11

DHT_Unified dht(DHTPIN, DHTTYPE); // DHT11 
sensor_t sensor; //DHT Sensor type
uint32_t sensorDelay; //Sensor delay in ms for more accurate readings

Servo servoMotor; //Servo motor declaration
int servoMotorPosition;

char ssid[] = SECRET_SSID;    // Network SSID
char pass[] = SECRET_PASS;    // Network Password
const char broker[] = MQTT_BROKER_ADDRESS; //Broker address
int port = MQTT_BROKER_PORT; //Broker port
const char deviceId[] = DEVICE_ID;
unsigned long getTime();

WiFiClient wifiClient;
BearSSLClient sslClient(wifiClient);
MqttClient mqttClient(sslClient);

bool willReset = false; //Used to reset the LWT if it's retained

String publishTopic  = "devices/" + String(deviceId) + "/messages/events/";
// String  publishTopic  = "devices/" + String(deviceId) + "messages/events/$.ct=application%2Fjson&$.ce=utf-8"; //Non base64 encoded
String  subscribeTopic  = "devices/" + String(deviceId) + "/messages/devicebound/#";

long previousMillis = 0;

// Used to status indicate on the MKRWifi for a visual indication of staus
void rgbColor(int r, int g, int b) {
        WiFiDrv::analogWrite(26, r);   //RED
        WiFiDrv::analogWrite(25, g); //GREEN
        WiFiDrv::analogWrite(27, b);   //BLUE
}

bool checkInterval(unsigned long interval) {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;  // save the last time
    return true;
  }
  return false;
}

void servoControl(int servoCommand) {
  Serial.println("Attaching");
  servoMotor.attach(SERVOPIN);
  Serial.println("Attached");
  delay(10);
  if(servoCommand == '0') {
    Serial.println("Engine position: 0");
    servoMotorPosition = 0;
    servoMotor.write(servoMotorPosition);
  };
  if(servoCommand == '1') {
    Serial.println("Engine position: 180");
    servoMotorPosition = 180;
    servoMotor.write(servoMotorPosition);
  };
  delay(1000);
  Serial.println("Detaching");
  servoMotor.detach();
  Serial.println("Detached");
}

void initCertificate()
{
  if (!ECCX08.begin()) 
  {
    Serial.println("No ECCX08 present!");
    while (1);
  }

  // Set a callback to get the current time - used to validate the servers certificate
  ArduinoBearSSL.onGetTime(getTime);

  // reconstruct the self signed cert
  ECCX08SelfSignedCert.beginReconstruction(0, 8);
  ECCX08SelfSignedCert.setCommonName(ECCX08.serialNumber());
  ECCX08SelfSignedCert.endReconstruction();

  // Set the ECCX08 slot to use for the private key and the accompanying public certificate for it
  sslClient.setEccSlot(0, ECCX08SelfSignedCert.bytes(), ECCX08SelfSignedCert.length());
  Serial.println("Certificate set");
}

void recievedMessage(int _) {
  char recievedCommand;
  while (mqttClient.available()) {
    recievedCommand = (char)mqttClient.read();
  };
  servoControl(recievedCommand);
}

void defineRGB(){
  WiFiDrv::pinMode(25, OUTPUT); //define GREEN LED
  WiFiDrv::pinMode(26, OUTPUT); //define RED LED
  WiFiDrv::pinMode(27, OUTPUT); //define BLUE LED
}

void defineMQTT(){
  mqttClient.setId(deviceId);
  
  String username = broker + String("/") + deviceId + String("/?api-version=2021-04-12");
  Serial.println(username);
  mqttClient.setUsernamePassword(username, "");
  mqttClient.onMessage(recievedMessage);

  // mqttClient.setUsernamePassword(MQTT_BROKER_USERNAME, MQTT_BROKER_PASSWORD);
  //mqttClient.setCleanSession(false);

  //Last will configuration
  // String willPayload = "offline";
  // mqttClient.beginWill("arduino/status", willPayload.length(), false, 1);
  // mqttClient.print(willPayload);
  // mqttClient.endWill();
}

void defineDHT11(){
  dht.begin();
  dht.temperature().getSensor(&sensor);
  dht.humidity().getSensor(&sensor);
  sensorDelay = sensor.min_delay / 1000;
}

void connectToWifi(){
  rgbColor(255, 0, 0); //Red
  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.print(".");
  }

  rgbColor(0, 255, 0); //Green 
  Serial.println("You're connected to the network");
  Serial.println();
}

void clearWill() {
  if (!willReset)
  {
    mqttClient.beginMessage("arduino/status", true, 1, false);
    mqttClient.print("");
    mqttClient.endMessage();
    willReset = true;
  }
}

void connectToMQTTBroker(){
  defineMQTT();
  Serial.print("Connecting to broker: ");
  Serial.println(broker);

  while (!mqttClient.connect(broker, port)) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }

  rgbColor(0, 0, 255); //Blue
  mqttClient.subscribe(subscribeTopic);
  Serial.println("Connected to the broker.");
  // clearWill();
}

unsigned long getTime() 
{ 
  return WiFi.getTime();
}


void setup() {
  
  defineRGB();
  defineMQTT();
  defineDHT11();

  Serial.begin(9600); //Serial BAUD rate on 9600
  while (!Serial) {
    ; // wait for serial port to connect
  }

  initCertificate();
  connectToWifi();

  // connectToMQTTBroker();
  // sendConnectedStatus();

}

///Reads telemetry data from DHT11
///Creates a JSON object with the data, outputs it to Serial Monitor
///Sends JSON to MQTT Broker
void readAndSendTelemetry(){
  float temperatureReading;
  float humidityReading;
  JsonDocument doc;
  sensors_event_t event;

  dht.temperature().getEvent(&event);
  temperatureReading = event.temperature;

  dht.humidity().getEvent(&event);
  humidityReading = event.relative_humidity;

  doc["temperature"] = temperatureReading;
  doc["humidity"] = humidityReading;

  char output[256];

  serializeJson(doc, Serial);
  Serial.println();
  serializeJson(doc, output);

  mqttClient.beginMessage(publishTopic);
  mqttClient.print(output);
  mqttClient.endMessage();
}
void loop() {
  if(WiFi.status() != WL_CONNECTED) {
    connectToWifi();
  };

  if(!mqttClient.connected()){
    rgbColor(255, 255, 0);
    connectToMQTTBroker();
  };

  mqttClient.poll();

  if(checkInterval(8000)) {
    readAndSendTelemetry();
  };
}