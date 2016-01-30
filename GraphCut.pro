TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    graph.cpp \
    maxflow.cpp \
    tools.cpp

HEADERS += \
    block.h \
    graph.h \
    instances.inc \
    tools.h
LIBS+=-L/usr/local/lib -lopencv_core -lopencv_highgui
INCLUDEPATH+=/usr/local/include
