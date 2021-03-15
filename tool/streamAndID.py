class IDManager():
    def __init__(self):
        self.id = ['E61605546-132', 'E61605498-101', 'E61605517-41']
        self.stream = ['rtsp://admin:1234abcd@192.168.1.132:554/Streaming/Channels/1',
                    'rtsp://admin:1234abcd@192.168.1.100:554/Streaming/Channels/1',
                    'rtsp://admin:1234abcd@192.168.1.41:554/Streaming/Channels/1'
                    ]
        self.idNum = len(self.id)

    def stream2ID(self, stream):
        bFind = False
        for i in range(self.idNum):
            if stream == self.stream[i]:
                id = self.id[i]
                bFind = True
                break
        if bFind:
            return id
        else:
            return False

    def id2Stream(self, id):
        bFind = False
        for i in range(self.idNum):
            if self.id == id:
                stream = self.stream[i]
                bFind = True
                break
        if bFind:
            return stream
        else:
            return False