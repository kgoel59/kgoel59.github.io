---
layout: post
title: "SOSE Week1"
date: 2024-06-24 08:30:00
description: Introduction
tags: projects learning uow
categories: learning
giscus_comments: true
featured: false
---

Service-Oriented Computing (SOC) is the computing
paradigm that utilizes services as fundamental elements
for developing applications/solutions

Services are self describing, platform-agnostic computational elements that
support rapid, low-cost composition of distributed applications

To build the service model, SOC relies on the Service Oriented Architecture
(SOA), which is a way of reorganizing software applications

Services are offered by service providers - organizations
that procure the service implementations, supply their service descriptions, and provide related technical and business support


 Clients of services can be other solutions or applications within an enterprise or clients outside the enterprise,
whether these are external applications, processes or customers/users.

 Technology neutral: they must be invocable through
standardized lowest common denominator technologies that are available to almost all IT environments.
This implies that the invocation mechanisms (protocols, descriptions and discovery mechanisms) should
comply with widely accepted standards.
• Loosely coupled: they must not require knowledge or
any internal structures or conventions (context) at the
client or service side.
• Support location transparency: services should have
their definitions and location information stored in a
repository such as UDDI and be accessible by a variety of clients that can locate and invoke the services
irrespective of their location.


Service comes in two flavour simple and composite

Composite services
involve assembling existing services that access and combine information and functions from possibly multiple ser-
vice providers.


 Accordingly, services help integrate applications
that were not written with the intent to be easily integrated
with other distributed applications and define architectures
and techniques to build new functionality while integrating
existing application functionality.


Service-based applications are developed as independent sets of interacting services offering

nor does it require pre determined agreements to be put into place before the use of an offered service is allow

 A web service is a specific kind of service that is
identified by a URI and exhibits the following characteristics:
• It exposes its features programmatically over the Internet using standard Internet languages and protocols,
and
• It can be implemented via a self-describing interface
based on open Internet standards (e.g., XML interfaces
which are published in a network-based repositories).


Interactions of web-services occur as SOAP calls carrying XML data content and the service descriptions of the
web-services are expressed using WSDL [15] as the common (XML-based) standard. WSDL is used to publish a
web service in terms of its ports (addresses implementing
this service), port types (the abstract definition of operations
and exchanges of messages), and bindings (the concrete
definition of which packaging and transportation protocols
such as SOAP are used to inter-connect two conversing end
points).


The UDDI [14] standard is a directory service
that contains service publications and enables web-service
clients to locate candidate services and discover their details.

The concept of software-as-a-service espoused by SOC
is revolutionary and appeared first with the ASP (Applications Service Provider) software mode

The SOC paradigm allows the software-as-a-service
concept to expand to include the delivery of complex business processes and transactions as a service, while permitting applications be constructed on the fly and services to be
reused everywhere and by anybody

To build integration-ready applications the service model
relies on the service-oriented architecture (SOA).

SOA is a
way of reorganizing a portfolio of previously siloed software applications and support infrastructure into an interconnected set of services, each accessible through standard
interfaces and messaging protocols

 The basic SOA defines an interaction between software agents as an exchange of messages
between service requesters (clients) and service providers.
Clients are software agents that request the execution of a
service. Providers are software agents that provide the service. Agents can be simultaneously both service clients and
provider


 Providers are responsible for desscription of the service(s) they provide. Clients must able
to find the description(s) of the services they require and
must be able to bind to them

The basic SOA is not an architecture only about services,
it is a relationship of three kinds of participants: the service
provider, the service discovery agency, and the service requestor (client). The interactions involve the publish, find
and bind operations

The service provider defines a service description of the service and publishes it to a client or service discovery agency
through which a service description is published and made
discoverable

Black box encapsulation inherits this feature from the principles of modularity in software
engineering, e.g., modules, objects and components

Reliable Design Without Implementation Knowledge: By defining the service specification, you can design and integrate services without needing to understand the internal implementations of the imported services. This abstraction allows you to compose services based on their interfaces and interactions, ensuring that you can design reliable service compositions.

