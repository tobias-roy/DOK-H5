∏# IOT H5 Data-technician

This is a repository of the assignments during my H5 course - specifically for the class IOT & Embedded 3.

## Shortcuts
[Notes](#notes) - this will lead you to all notes

[Mini Svendeprøve](#mini-svendeprøve)

[MQTT Basics](#mqtt)

[MQTT Questions](#mqtt-questions)

[MKR MQTT Group Project](#project)

# Worklog
[Day 1 - Introduction day](#day-1---introduction-day)

[Day 2 - MQTT Continued](#day-2---mqtt-continued)

[Day 3 - MQTT Continued](#day-3---mqtt-continued)

[Day 4 - Project Startup](#day-4---project-startup)

[Day 5 - Project Ending](#day-5---project-ending)

[Day 6 - Azure IOT Hub](#day-6---azure-iot-hub)

## Day 1 - Introduction day
Downloaded and installed [PlatformIO Core and PlatformIO extension](https://platformio.org/) for vscode.

Assignment 2.1 - booting up a [MKR WIFI 1010](https://docs.arduino.cc/hardware/mkr-wifi-1010/)

Assignment 2.2 - Added git verison control 

Assignment 4.1 - Connecting to wifi with [WiFiNINA](https://docs.arduino.cc/tutorials/communication/wifi-nina-examples/)

Assignment 4.2 - Sending a package via [WiFi & SSL](https://docs.arduino.cc/tutorials/communication/wifi-nina-examples/#wifinina-wifi-ssl-client) and getting a response

Went over a lot of stuff in regards to MQTT basics.

## Day 2 - MQTT Continued
Quick sum up from yesterday

Assignment 5.3 - MQTT and shiftr.io with MQTTX client - here we created clients with MQTTX and used the shifrt.io cloud service as a broker to get a map of what MQTT messages and topics we broadcasted.

Assignment 6.1 - Publisher/Subscriber metoder igennem

Broker credentials for SHIFTR: mqtt://h5dok:AjZSoheCb8EAOS31@h5dok.cloud.shiftr.io

Assignment 6.2 - Fra en MQTT client skal vi sende en streng: ON / OFF - det skal modtages på boardet og så skal den tænde eller slukke på baggrund af hvad strengen indeholder og servoen skal køre

Assignment 6.3 - Connecting a DHT11 to the board and transmitting data via MQTT from the board to our interface.

Assignment 6.4 - HiveMQ Cluster Passwords 'TestCluster1'

Assignment 6.5 - Basicly the same as the previously 6 assignments just change the connection strings.

Assignment 6.6 - Controlling the servo via HiveMQ is also the same as the previous assignment


## Day 3 - MQTT Continued

Continued with the assignments from yesterday

Assignment 7.1 - QoS and CleanSession on PC 

Assignment 7.2 - QoS and CleanSession on Arduino

Assignment 7.3 - Retained Messages

Assignment 7.4 - Retained Messages with Arduino

Assignment 7.5 & 7.6 - Last Will and Testament 

Assignment 7.7 Client Takeover

Quiz


## Day 4 - Project startup
Quick repetitions

Project startup

[MKR MQTT Group Project](#project)


## Day 5 - Project ending

Project ending

[MKR MQTT Group Project](#project)


## Day 6 - Azure IOT Hub

Theory about Azure

Short case introduction for the *Mini svendeprøve*

Working on assignemnts for creating azure IoT stuff.

Created PowerBI solutions

## Day 7 - Azure IOT Hub

Continuing working Azure IoT hub.



## Svendeprøve forløb
Gruppe fremlæggelse 12 minutter - fungere som en demonstration af det produkt man har lavet, det fungere som en salgsfremstilling. Alle skal have taletid etc.

Individuelle fremlæggelser 40 minutter inkl. votering (5min), 35min præsentation inkl spørgsmål. - der udvælges et emne fra projektet at snakke ud fra. Det er vigtigt at det tager udgangspunkt i projektet. Regn med 30 minutter og hav en lille ekstra ting man kan præsentere og forvent at blive afbrudt i det.

Det nye er at den personlige fremlæggelse får fjernet nogle minutter - 3 minutter fra hver præsentation ca.

Det personlige emne eller teknologi der vælges kan være f.eks Kryptering, Authentication & Authorization, SignalR

Kom ned på protokol niveau - brug evt. wireshark til visualisering af 0-1 bits i trafikken.

Byg udfordringer/problemer og løsninger ind i fremlæggelsen for at visualisere tilgangen til at imødekomme den "korrekte" løsning.

Det Egon syntes er spændende er: Kombinationen af openAI i dit eget software, services osv.

Mini svendeprøven er en todelt case.

# Notes
### General

= at the end of a string typically indicates the format is BASE64

### MQTT 
The lightweigh data transfer protocol. Developed for transfering machine telemetry with minimal battery loss and minimal bandwidth. It's data agnostic because it transfers binary data. The MQTT requirements are the requirements we also have to IoT today: 

- Simple implementation
- Quailty of Service data delivery (QoS)
- Lightweight and bandwidth efficient
- Data agnostic
- Continuous session awareness

Today we use MQTT 3.1.1 as the industry standard which is also ISO certified.

### MQTT Characteristics

- Binary
- Efficient, the smallest package is 2 bytes
- Bi-Directional, it works both ways
- Data-agnostic since it's binary 
- Scaleable, you can run more than 10 million devices on the same installation
- Built for push communication, with the broker principle
- Suitable for constrained devices, MQTT requires so little device requirements are very low.

### MQTT is build on top of TCP

- MQTT reuires TCP/IP
- Persistent TCP connections (numbered packages and acknowledgements)
- Heartbeat mechanism is built into it which will re-establish a broken connection
- Security on transport level (TLS)

### MQTT Publish / Subscribe pattern (PUB/SUB for short)

- Client / Server protocol - Client Sends a request to the server, server sends a response.
- The MQTT way of doing this is with the publish/subscribe pattern. MQTT Clients publish to a MQTT broker, the broker then sends data to the interested subscribers which are also MQTT clients. These clients can also be publishers since is bi-directional.

- The broker is decoupled in regards to consuming/delivering data, it will stack data in a que.

- The broker is the Single Point of Faliure in MQTT, if you choose to work with this profesionnaly the broker software should support clusters.

- MQTT components consists of Clients (Subscribing and publishing) and MQTT Brokers

### Connection flow
1. Client establishes TCP to the broker
2. The MQTT Connectionflow starts.
3. Client sends a connect packet to the broker which consists of, clientId, username/password base or token, lastwill, testament and keepalive information, clean session flag - this indicates if the broker should forget og remember the client and session upon reconnet/disconnect.
4. After the connect packet is recieved, the broker sends a CONNACK.
5. CONNACK constist of sessionPresent and returncode, wether or not to indicate if we are allowed to have a connection.

#### Publish
- The client creates a MQTT PUBLISH packet. It consists of a packetId, topicName (this is very important!), qos level, retainFlag, payload field (this is all relevant information, ie the data) and a dupFlag.
- You can send up to 256mb, usually the packets are below 1mb.
- The publisher sends the packet to the broker which distributes it to the subscriber clients.

#### Subscribe
- A client sends a subscribe packet to the broker to indicate it wants to subscribe to a topic.
- It consists of a pakcetId, qos1 topic, qos2 topic etc.
- It sends this packet to the broker which responds with a SUBACK packet.
- The client can also send a UNSUBSCRIBE packet, which consists of the topics you want to ubsubscribe from.
- The broker then responds with a UNSUBACK

### Sending MQTT 
[Microsofts documentation on the topic](https://learn.microsoft.com/en-us/azure/iot/iot-mqtt-connect-to-iot-hub#use-the-mqtt-protocol-directly-from-a-device)

#Define the secret broker the hub.azure-devices.net"

#define secret_device_id "match-device-on-cloud"

The device needs a self-signet x509 certificate it uses a primary thumbprint and a secondary thumbprint for the auth.

The privatekey is stored on the board in the cryptographic chip in front of the WiFi module. The ATECC508A chip.
This is the certificate for Board 4 in the classroom for WifiNINA v.1.5.0:

````
-----BEGIN CERTIFICATE-----

MIIBLDCB06ADAgECAgEBMAoGCCqGSM49BAMCMB0xGzAZBgNVBAMTEjAxMjMzMzhERDRDQzVFNjdF

RTAgFw0yMzAzMjUxMjAwMDBaGA8yMDU0MDMyNTEyMDAwMFowHTEbMBkGA1UEAxMSMDEyMzMzOERE

NENDNUU2N0VFMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEc+i0fakUywH+ffuc3wBeaM5vBlhA

GTwFESFZLZkcS9vL/4J/A3dayFR5IqxLwjBnA2kJd+DnNCl5m4j3WlLozKMCMAAwCgYIKoZIzj0E

AwIDSAAwRQIhAKjXzeshH7VLYP4jXm+oToLepXyahQt8HBuaB1C1u4vxAiA+Yeju8VxkElCWD9Hw

C68yWYsczteajrF1X2WdG7fyKw==

-----END CERTIFICATE----- 
````
This is the __Certificate thumbprint__

SHA1: __1568a9e564097b18aa6b9e9c685fca35812ce291__

----
We will need the base64 encoding at some point in the assignemnt

### Best practices
### Topics
Topic is a UTF8-String, the topic consits of multiple levels.
```USA/Califonia/SanFrancisco``` 
This is a topic with 3 levels divided by the delimeter '/'. Topics are case sensitive. Clients can publish to any topic and thus they don't need to be pre-defined.

The '+' sign is used as a wildcard. That indicates that the subscription you want should match anything on a speicifc Topic level.

The '#' wildcard indicates that everything after the # should match the subscriptions.

Best practices for Topics
- Never use leading forward slash
- Never use spaces
- ONLY ASCII characters
- Keep topics short and concise
- Embed a unique identifier or client ID in a topic.
- Never subscribe to root #

### QoS
QoS ensures guaranties between the publisher and subscriber. 
- QoS 0, messages sent once then lost (At most once delivery) - used when you don't need to que any messages, where loss is acceptable. This could be used for sending metrics constantly.
- QoS 1, messages repeatedly sent until an ACK is recieved from the destination duplicate possibility (At least once delivery) - usual default, great tradeoff between bandwidth and delivery guarantee.
- QoS 2, messages repeatedly sent until an ACK is recieved, without duplicate (Exactly once delivery) - When you wan't exactly one and only one.

### Persistent Sessions and Queueing (cleanSession Flag)
The Connect package carries the 'cleanSession' flag. If this is false you tell the broker you want a persistent connection that the broker will remember.

The broker will remember the Session data (clientID) - Subscriptions from the client - Unacknowledgted QoS messages - Queued messages

Messages are queued per client - The broker queues all QoS 1 and 2 messages when a persistent session client is offline

Best practices

 - A pæersistent session is recommended for subscribe and subscribe/publish clients. 

 - Clean session TRUE is recommended when a client only needs to publish and message loss is acceptable.
 - Persistent Session is recommended when subscribers must not miss messages and the broker should store subscription information

Retained messages will be retained for each new subscriber to a specific topic.


### Last Will
If a client looses network connection a message will be sent out according to the lastWill setup in the CONNECT packet.

The lastWillTopic is the topic the lastWillMessage will be sent on. LastWillQos defines the QoS of the message and the lastWillRetain is set to wether or not new subscribers to the topic will get the retained message.

Cases for LWT

- If the client fails to send a packet within the Keep Alive period the LWT will be used.

- If the client does not send a DISCONNECT packet.

- If the broker closes the connection (protocol errors)

Keepalive max time is 18 Hrs

### Client takeover

Client takeover will happen if a client established a connection with an already used client ID. The broker will in this case close the old connection and establish a new one.

Best practices

- Use uniqeu CLientIds
- Authenticate clients to prevent unwanted Client Takeovers


### Packets
|CONNECT||
|---|---|
|clientId|"client1"|
|cleanSession|true|
|username|"hans"|
|password|"letmein"|
|lastWillTopic|"/hans/will"|
|lastWillQos|2|
|lastWillMessage|"unexpected exit"|
|lastWillRetain|false|
|keepAlive|60|
---

|CONNACK||
|---|---|
|sessionPresent|true|
|returnCode|0|

---

|Publish||
|---|---|
|clientId|"client1"|
|topicName|"client/status"|
|payloadFormatIndicator|1|
|contentType|SolarPanelSchemaV1.0|
|qos|1|
|retainFlag|true|
|payload|"online"|

---

|SUBSCRIBE||
|---|---|
|packetId|2313|
|qos1|1|
|topic1|"topic/1"|
|qos2|0|
|topic2|"topic/2"|
|...|....|

---

|SUBACK||
|---|---|
|packetId|2313|
|returnCode 1|2|
|returnCode 2|0|
|...|....|

---

|UNSUBSCRIBE||
|---|---|
|packetId|2313|
|topic1|"topic/1"|
|topic2|"topic/2"|
|...|....|

---

|UNSUBACK||
|---|---|
|packetId|2313|

---

## MQTT Questions

### Questions 1 - 3
1. 1999
2. MQTT doesnt stand for anything today, but it used to stand for Message Queuing Telemetry Transport
3. OASIS and ISO
4. The industry standard is 3.1.1, the newest is 5 from 7 March 2019
5. MQTT Clients publish to a MQTT broker, the broker then sends data to the interested subscribers which are also MQTT clients. These clients can also be publishers since is bi-directional. It's a spiderweb of interconnection.
6. Postnord is a broker in Denmark
7. Space, time and syncronization decoupling.
8. No, it uses Publish/Subscribe

### Questions 4 - 6
1. PUB/SUB is very scalable and supports decoupling which is highly persistent in regards to loosing/gaining connections and keeping connectivity. Requires less resources than HTTP, it's binary (Data-agnostic).
2. MQTT is the application layer, TLS in the presentation Layer, TCP in the transport layer, IP in the network layer, WIFI/Ethernet...etc... in the Datalink and physical layer.
3. See [Connection flow](#connection-flow)
4. See [Connection flow](#connection-flow)


# Project

DHT11 -> MKR -> HiveMQ data (PUB)
HiveMQ -> MKR -> Servo (SUB)

REST WebApi integration with MQTT - getting and posting to and from HiveMQ

The rest API should subscribe/get data from the MQTT broker, temperature etc. from DHT11

The rest API should be able to post to make the servo motor move

HiveMQ Api for information of Subscribers and Publishers: https://docs.hivemq.com/hivemq/latest/rest-api/index.html

InfluxDB



## Project start

- Repository share, HiveMQ Broker
- Access Management HiveMQ
- USR/PW - RestAPI - restapiAdmin1234
- USR/PW - MKRTEL - mkrAdmin1234

---

- API, bliver Minimal API - Andrias
- InfluxDB, Dennis startup
- Mkr kode integration, Tobias

--- 


# Azure IOT Hub

![Microsoft Cloud Models](readmeImages/cloudmodels.png)

IAAS - Infrastructure as a Service

PAAS - Platform as a Service

SAAS - Software as a Service

The free tier we use for our IoT hub has 8.000 messages per hub per day.

### Device Twin



### Iot hub
<img src="readmeImages/iothub.png" width="auto" height="250"/>

Message routing is comparable to the *topics* in MQTT. In Azure we have a lot of functionality that we can use without the need for actual programming.  

## Routing Query Syntax

<img src="readmeImages/routingquery.png" width="auto" height="250">



## .NET Virtual IoT Device (Device Twins)
C# Device clients interaction with Azure

<img src="readmeImages/devicetwincsharp.png" width="auto" height="250">

Device twin contains properties for *desired* values and *reported* values. If your desired value for temperature is 22 degress but the reaported temperature is 18 degrees, this could cause other IoT devices to boot and produce heat.

Device twins can contain tags - tags can be used to define what specific device we are connected to. It can be used to specify Country/Region/City/Building etc..

## IoT Hub Partitions

<img src="readmeImages/partitions.png" width="auto" height="250">


We can have a maximum amount of 32 Partitions per IoT Hub. 

<img src="readmeImages/consumergroups.png" width="auto" height="320">

Consumer groups are used by applications to pull data from the IoT Hub.

Each consumer group has its own settings for state, position and offset for each partition. The "Checkpoints" marked act as an indicator to show where in the datastream each partition is at the moment. Look at it as a Super Mario checkpoint.

## Event Processor Host
<img src="readmeImages/eventprocessor1.png" width="auto" height="200">
<img src="readmeImages/eventprocessor2.png" width="auto" height="200">
<img src="readmeImages/eventprocessor3.png" width="auto" height="200">

Event processor Host scaling can be viewed as a form of cluster.

## AMQP Protocol
Advanced Message Queuing Protocol, created in 2003 by J.P. Morgan Chase, for systems that require a high level of reliability and functionality.

A PEER-to-PEER protocol between Producer and Consumer. Messages are sent to exchanges, which then route the messages to apporpriate queues based on rulse called bindings. The consumer then retrieves the message from the queue. This model allows for complex routing and distribution strategies.

## Messaging to devices
<img src="readmeImages/m2d.png" width="auto" height="250" />

Cloud to device message is a text message being sent to devices from the Cloud. It works for offline devices because a que per device is being created.

Direct Method is a function on the device being activated by the cloud. A use case could be to activate haptic feedback.

Device Twins is used for long running commands where things should "happen" like the temperature regulator case.

## DPS - Device Provisioning Service

<img src="readmeImages/dps.png" width="auto" height="250" />

The diagram shows how DPS functions. The DPS is sort of a handler for devices. It will aid in the creation of devices for the HUB.

The device needs to know three things. Registration ID, Primary key and 

## IIoT
Industrial Internet of Things - 

## Mini svendeprøve
We are going to create a prototype to a wristband used in a themepark.

The wristband contains 3 buttons and a display.
- Happy Smiley - Sends JSON
- Sad Smiley - Sends JSON
- Emergency button - Sends JSON
- Display - Recieves JSON for function printMsg()

```
//Happy and sad
{
    "fb": 1, //1 for Happy 0 for sad
    "gps": "lat, long"
}
```

```
// Alert
{
    "alert": "1",
    "gps":"lat, long"
}
```

```
//Display printMsg
{
    "msg": "This is a message!"
}
```

On the board we have a Heartbeat function. The API will send a "Request heartbeat" request to a topic that all devices subsribe too - the device will then resbond with a JSON object: 
```
{
    "bat": 23,
    "gps": "lat, long"
}
```

## Topics

```
//From top to bottom level
wristband/

wristband/{deviceid}

wristband/{deviceid}/heartbeat/out //C2D
wristband/{deviceid}/heartbeat/in //D2C

wristband/{deviceid}/feedback //D2C

wristband/{deviceid}/emergency //D2C

wristband/announcement //C2D for messaging to all devices
wristband/announcement/{deviceid} //C2D for messaging to specific devices
```
## Pre-requisites
We will implement:
- Github Actions
- Api hosting on Azure?



### Assignments

Andrias - WebApi

Emma - WebApi, Kravspecifikation

Tobias - UI blazor WASM

Dennis - Makerboard




The wristband should be able to:
- Track guests location
- Real-time communication (Feedback and park notifications) we want to be able to send anything from the "central" to all park guests.

We will be using HiveMQ - Make use of the HiveMQ API for tracking devices.

The pipeline consists of 4 steps.
 
<img src="readmeImages/globo.png" width="auto" height="250" />

As little as possible should be handled on the board itself.




Checklist (On-going TODO):
- Documentation: Create a naming convention for Device ID's
- Documentation: Security measures and steps
- Documentation: 