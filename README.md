# Filtering of recognisable duplicate tickets

The purpose of this project is to find a subset of Incident Management (IM) tickets recognized as duplicate ticket of another IM ticket. These tickets are grouped in different ways, according several criterias.

## Wiki

The tickets are univocaly identified by a ticket number (might be also called ticket ID or ticket identifier) and the ticket type (if it is a Incident ticket IM, a Problem ticket PM, a Request of Fulfillment ticket RF or a Service Desk ticket SD). This project focus only on Incident tickets IM.

![Ticket structure: it is composed by a ticket type and the ticket number](https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/ticket_structure.png?raw=true)

The ticket type is fixed and it is the same for ticket belonging to the same type. While the numeric part is incremental: every time a new Incident is identified a new ticket is raised with the numeric part increased by one.

Often, different tickets are raised for the same Incident but in a diverse moment in time: the first raised ticket is the __origin__, the second ticket is the __duplicate__ of the first ticket.

![Three IM tickets are raised: IM01 and IM03 are referring to the same Incident, so IM01 is the origin and IM02 is the duplicate (of the ticket IM01). IM02 is referring to a different Incident.](https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/connection_duplicate_origin_tickets.png?raw=true)

When a ticket is recognized as a duplicate ticket (of a oldest ticket), the duplication is defined in the Solution field of the ticket itself and the duplicate ticket is closed.

## Project Structure

- core (folder)
    - CompareCodes (.py)
    - FileHandler (.py)
    - OutputHandler (.py)
    - itsm_im_tickets_constrains (.py)
    - im_stars_representation (.py)
- exceptions (folder)
    - RTDException (.py)
    - RTDWarning (.py)
    - FileHandlerException (.py)
    - FileHandlerWarning (.py)
    - FileNotExistException (.py)
    - NoValidExtensionException (.py)
    - HeaderNotExistWarning (.py)
- images (folder)
- sources (folder)
- tests (folder)
    - FileHandlerTest (py)
- main (.py)

> NOTE:
>
> The sources folder contains the excel file and it is not present in the repo for confidentiallity, please create 
> the `sources` folder and use it to save your excel file.

## Requirements

The dataset containing the IM tickets, the following headers may be required:
- Ticket Code - Unique identifier of a IM ticket
- Solution - Free text box where it is described how the incident was resolved
- Open Time - Date when the ticket was opened
- Contact - The operator (or agent) who opened the ticket
- Resolved Time - Date when the ticket was resolved
- Closed By - The operator (or agent) who closed the ticket

## The Program
The purpose of this project is to find a subset of Incident Management (IM) tickets recognized as duplicate ticket of another IM ticket. These tickets are grouped in different ways, according several criterias.

### Find the subset of IM Tickets
The __Solution__ field of a ticket is analysed by means of Regular Expression looking for terms like _"duplicate"_, _"duplication"_ or similar, in combination with a ticket code: the origin ticket. These tickets are duplicate tickets, identified by an operator.

### Regex and Ticket cases
The Solution field is a free-text box where the operator can write whatever he/she wants. The text is analyzed to understand if the ticket is a duplicate or not. However, since the operator has a certain degree of freedom, the text may contains one or multiple duplicate terms and ticket codes.

For that reason, the tickets are also categorized in cases, as it is shown in the following picture.


<img src="https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/Cases_of_recognized_duplicate_tickets.png?raw=true"  width="450" height="500">

Five cases are identified, according the different presences of the key terms. In the program, cases: 1, 2, 4 are collected together for the presence of a single ticket code in the text.

### Pair of Tickets Violating ITSM rules
Let's consider two tickets for the same Incident, one is the origin and the other one is the duplicate, they can be considered as a pair of tickets (duplicate, origin).

Pairs of tickets have to follow some rules, rules decided by the ITSM system.

- Pair of tickets are of the same ticket type
- Origin and duplicate have not the same ticket code
- Numeric part of the origin ticket is lower than the numeric part of the duplicate ticket
- If duplicate ticket is pointing to the origin ticket, the origin ticket cannot point to the duplicate ticket (cycles are not allowed)

If a pair of ticket is violating any of this constrains, the program takes note of the violation.

### Origin and Duplicates - Star Diagram
If for an Incident we have two tickets, in that case we have a pair. But if multiple tickets are raised for the same Incident, than we have a Star configuration, i.e. one origin ticket and multiple tickets defined as duplicate tickets (of the origin tickets). The program identified the following type of stars according the depth of the diagram, i.e. the number of edges connecting the start-origin to the furthest duplicate ticket.

<img src="https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/path_length_1.png?raw=true"  width="350" height="300">

In this case the star diagram has just one single duplicate ticket, we can understand that a pair of ticket (duplicate, origin) is the simplest representation of a star diagram.

<img src="https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/path_length_2.png?raw=true"  width="350" height="300">

In this case the star diagram is composed by a set of duplicate tickets. In this case the star shape is more visible and the star diagram has some tickets directly pointing to the origin and others that are pointing to duplicate tickets: some tickets are creating a chain of tickets in which a Incident ticket identify another duplicate ticket as origin. 

<img src="https://github.com/wilsonjefferson/DSSC_IM-DUPLICATES-HUNTER/blob/main/images/path_length_3.png?raw=true"  width="350" height="300">

This is a more complex case.

_Note:_
Please, refer to the images folder, here you can find all the picture.
