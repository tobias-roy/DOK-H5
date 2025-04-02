# IOT H5 Data-technician

This is a repository of the assignments during my H5 course - specifically for the class IOT & Embedded 3.

## Shortcuts
[Notes](#notes) - this will lead you to all notes

[MQTT Basics](#mqtt)

[MQTT Questions](#mqtt-questions)

# Worklog
[Day 1 - Introduction day](#day-1---introduction-day)

[Day 2 - MQTT Continued](#day-2---mqtt-continued)

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


## Day 3



# Notes
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
- QoS 0, messages sent once then lost
- QoS 1, messages repeatedly sent until an ACK is recieved from the destination duplicate possibility
- QoS 2, messages repeatedly sent until an ACK is recieved, without duplicate



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

