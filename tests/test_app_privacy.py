from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

import app as app_module


class PrivateMediaTests(unittest.TestCase):
    def setUp(self):
        self.storage = TemporaryDirectory()
        self.upload_folder = patch.object(
            app_module, 'UPLOAD_FOLDER', self.storage.name
        )
        self.upload_folder.start()
        self.addCleanup(self.upload_folder.stop)
        self.addCleanup(self.storage.cleanup)
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()
        self.other_client = app_module.app.test_client()

    def upload_images(self):
        person = BytesIO()
        Image.new('RGB', (32, 48), 'blue').save(person, format='PNG')
        clothing = BytesIO()
        Image.new('RGBA', (32, 32), 'red').save(clothing, format='PNG')
        return self.client.post(
            '/upload',
            data={
                'person_image': (BytesIO(person.getvalue()), 'person.png'),
                'clothing_image': (
                    BytesIO(clothing.getvalue()), 'clothing.png'
                ),
            },
            content_type='multipart/form-data',
        )

    def test_uploaded_images_are_private_to_the_browser_session(self):
        self.assertEqual(self.upload_images().status_code, 302)
        self.assertEqual(self.client.get('/preview').status_code, 200)
        self.assertEqual(self.client.get('/viewer').status_code, 200)
        self.assertEqual(self.client.get('/avatar').status_code, 200)
        response = self.client.get('/media/uploads/person.jpg')
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(
            self.client.get('/static/uploads/person.jpg').status_code, 404
        )
        self.assertEqual(
            self.other_client.get('/media/uploads/person.jpg').status_code,
            404,
        )
        self.assertTrue(self.client.get('/recommend').json['success'])

    def test_result_media_is_private_to_the_browser_session(self):
        self.upload_images()

        def fake_pipeline(person_path, clothing_path, output_path,
                          debug_output_dir=None):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (32, 48), 'green').save(output_path)
            return True, output_path

        with patch.object(app_module, 'run_tryon_pipeline', fake_pipeline):
            result = self.client.post('/tryon')

        self.assertTrue(result.json['success'])
        response = self.client.get(result.json['result_url'])
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(
            self.other_client.get(result.json['result_url']).status_code, 404
        )
        self.upload_images()
        self.assertEqual(self.client.get('/result').status_code, 302)
        self.assertEqual(self.client.get(result.json['result_url']).status_code, 404)


if __name__ == '__main__':
    unittest.main()
