import React, {useEffect} from 'react';
import styles from './InspectorMain.module.scss';
import { useNavigate } from 'react-router-dom';
import { apiGetProjects } from '../../../api/api';
import AllObjects from '../../../components/AllObjects/AllObjects';
import linksStore from '../../../store/linksStore';
import store from '../../../store/store';

const InspectorMain = () => {
	const { linkGetProjects} = linksStore;
	const { allObjects, setAllObjects } = store;
	const navigate = useNavigate();

	useEffect(() => {
		getObjects(linkGetProjects);
	}, []);

	const getObjects = (url = '') => {
		apiGetProjects(url).then(({data, error}) => {
			setAllObjects(data.map(({samolet_pk, name, card_image
			}) => {
				return {
					id: samolet_pk,
					name,
					photo: card_image
				}
			}));
			// console.log(data);
			console.log(error);
		})
	};

	const selectObject = (id) => {
		navigate(`/employee/${id}`);
	};

	return (
		<AllObjects
			data={allObjects}
			title={'Выберите ЖК для обхода'} 
			handleSelectObject={(id) => selectObject(id)}
		/>
	);
};

export default InspectorMain;
