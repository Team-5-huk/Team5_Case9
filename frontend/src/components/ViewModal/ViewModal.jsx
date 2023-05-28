import React from 'react';
import styles from './ViewModal.module.scss';
import { Modal } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const ViewModal = ({ title, isModal, closeModal, children }) => {
	return (
		<Modal open={isModal} onClose={closeModal} closeAfterTransition>
			<div className={styles.container}>
				<div className={styles.title}>
					{title}
					<CloseIcon style={{ cursor: 'pointer' }} onClick={closeModal} />
				</div>

				{children}
			</div>
		</Modal>
	);
};

export default ViewModal;
