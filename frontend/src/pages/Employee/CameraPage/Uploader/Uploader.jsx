import React from 'react';
import store from '../../../../store/store';
import { observer } from 'mobx-react-lite';

const Uploader = observer(() => {
	const { files } = store;
    console.log(files);
	return (
		<div>
			Загрузки
			{files.map((file) => (
				<div className="upload-file">
					<div className="upload-file__header">
						<div className="upload-file__name">{file.name}</div>
						<div className="upload-file__remove" >
							X
						</div>
					</div>
					<div className="upload-file__progress-bar">
						<div className="upload-file__upload-bar" style={{ width: file.progress + '%' }} />
						<div className="upload-file__percent">{file.progress}%</div>
					</div>
				</div>
			))}
		</div>
	);
});

export default Uploader;
