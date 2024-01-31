import unittest
from m3api.m3api import to_dclass, to_dict, CoerceError, MIResult, MIMetadata, MIRecord, MINameValue, MIField 


class TestM3Api(unittest.TestCase):
    def setUp(self):
        self.fields = [
            MIField(name='Field1', type='N', length='1', description='1 Numeric Integer'),
            MIField(name='Field2', type='N', length='3', description='2 Numeric Float'),
            MIField(name='Field3', type='D', length='6', description='3 Date'),
            MIField(name='Field4', type='A', length='9', description='4 Alpha')
        ]
        
        self.records = [
            MIRecord(
                RowIndex=1, NameValue=[
                    MINameValue(Name='Field1', Value='1'),
                    MINameValue(Name='Field2', Value='1.0'),
                    MINameValue(Name='Field3', Value='20240101'),
                    MINameValue(Name='Field4', Value='OK')
                ]
            ),
            MIRecord(
                RowIndex=2, NameValue=[
                    MINameValue(Name='Field1', Value='2.0 '),
                    MINameValue(Name='Field2', Value=' 2'),
                    MINameValue(Name='Field3', Value=' 20240102'),
                    MINameValue(Name='Field4', Value='   OK with leading and trailing spaces    ')
                ]
            ),
            MIRecord(
                RowIndex=3, NameValue=[
                    MINameValue(Name='Field1', Value='THREE'),
                    MINameValue(Name='Field2', Value='3.0'),
                    MINameValue(Name='Field3', Value='20240103'),
                    MINameValue(Name='Field4', Value='FAIL')
                ]
            ),
            MIRecord(
                RowIndex=4, NameValue=[
                    MINameValue(Name='Field1', Value='4'),
                    MINameValue(Name='Field2', Value='FOUR'),
                    MINameValue(Name='Field3', Value='20240104'),
                    MINameValue(Name='Field4', Value='FAIL')
                ]
            )   
        ]

    def test_to_dict_no_coerce(self):
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records
        )

        #check that no coerce error is raised
        results = list(to_dict(mi_result, coerce=False))

        #check that all records exits
        self.assertEqual(len(results), 4)

        #check that all fields are included
        self.assertTrue(all(len(r) == 4 for r in results))


    def test_to_dict_with_coerce_good_data(self):
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records[:2]
        )
        
        #check that the two first records coerces without errors
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records[:2]
        )

        coerce_dict = list(to_dict(mi_result, coerce=True))

        #check that the conversion of values are correct
        self.assertEqual(coerce_dict[1]['Field1'], 2.0)
        self.assertEqual(coerce_dict[1]['Field2'], 2)
        self.assertEqual(coerce_dict[1]['Field3'], 20240102)
        self.assertEqual(coerce_dict[1]['Field4'], 'OK with leading and trailing spaces')


    def test_to_dict_coerce_bad_data(self):
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records
        )
        
        with self.assertRaises(CoerceError):
            list(to_dict(mi_result, coerce=True))

        #check that each of the bad records causes a fail
        for record in self.records:
            mi_result['MIRecord'] = [record]

            if 'FAIL' in record['NameValue'][3]['Value']:
                with self.assertRaises(CoerceError):
                    list(to_dict(mi_result, coerce=True))

    
    def test_to_dclass_good_data(self):
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records[:2]
        )

        #check that converion does not cause errors for good records
        list(to_dclass(mi_result))


    def test_to_dclass_bad_data(self):
        mi_result = MIResult(
            Program='m3api_program',
            Transaction='m3api_transaction',
            Metadata=MIMetadata(Field=self.fields),
            MIRecord=self.records
        )

        #check that each of the bad records causes a fail
        for record in self.records:
            mi_result['MIRecord'] = [record]

            if 'FAIL' in record['NameValue'][3]['Value']:
                with self.assertRaises(CoerceError):
                    list(to_dclass(mi_result))


if __name__ == '__main__':
    unittest.main()
