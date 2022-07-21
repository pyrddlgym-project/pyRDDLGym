from Parser import parser

#DOMAIN = 'power_unit_commitment.rddl'
DOMAIN = 'ThiagosReservoir.rddl'

def main():
    domain = ""
    with open('RDDL/'+DOMAIN) as file:
        domain = file.read()

    MyRDDLParser = parser.RDDLParser()
    MyRDDLParser.build()
    rddl = MyRDDLParser.parse(domain)
    print(rddl)

    print("test")





if __name__ == "__main__":
    main()



